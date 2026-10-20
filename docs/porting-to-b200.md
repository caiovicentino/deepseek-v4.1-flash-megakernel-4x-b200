# Porting `sgl-nvidia-mega` from GB300 (SM103) to B200 (SM100)

All of this was done on 2026-10-19 without touching the GPUs, which were serving production. Each check below can be repeated with the tools in this repo.

## 1. What tied the plugin to GB300

| Place | GB300 assumption | Kernel constant or guard? |
|---|---|---|
| `deepseek_v41/spec.py` | `isas=frozenset({"sm_103"})` | Guard: the framework refuses any other ISA |
| `deepseek_v41/spec.py` | `resident_blocks=152` (GB300's SM count) | Guard: the framework refuses GPUs with fewer SMs (meant for partitioned GPUs) |
| `sm103/mxfp8_ll.py` | Pinned GEMM tactics for `(103, N, K)` | Tuning: other GPUs get FlashInfer's default tactic |
| folder name `sm103/` | none | Naming only |

Nothing else in the plugin's Python checks the compute capability.

## 2. Do the kernels build for SM100?

The plugin JIT-compiles its CUDA sources with NVRTC for `sm_{major}{minor}a` of the GPU it finds. [`tools/check_nvrtc_sm100.py`](../tools/check_nvrtc_sm100.py) compiles every `.cu` with the plugin's exact options and `-D` defines for both `sm_103a` (control) and `sm_100a`, using NVRTC 13.0:

- route, route_wide, gemm1, gemm2, gate_topk, gate_topk_draft, the two-shot all-reduce with mHC (12 valid of 16 combinations), the Markov draft (4) and the fused `wo_a` (12): **34/34 valid variants compile for `sm_100a`**, and their cubins are the same size as the `sm_103a` ones.
- 4 variants fail on **both** targets. They are combinations the source forbids with `static_assert(!kQuant || kNorm)`, a harness artifact.

The CuTe DSL kernels (`all_reduce_mhc`, `wo_b_publish`, `mxfp8_ll`) compile only at runtime. They use `cutlass.utils.blackwell_helpers` (SM100-family helpers), and their twins in vLLM PR #61868 run in vLLM's CI on 4× B200. FlashInfer 0.7.x lists `cutedsl_low_latency` (the GEMM `mxfp8_ll` calls) as SM100/SM103.

## 3. Does anything need more than 148 co-resident blocks?

From `cuobjdump -res-usage` on the `sm_100a` cubins:

| Kernel | Registers × threads | Blocks / SM | Max grid | Waits on other blocks? |
|---|---|---|---|---|
| `two_shot_mhc` (rows 49–384) | 64–72 × 128 | 7 (by registers) | 384 × 5 = 1,920 | Yes, on other GPUs' fragments, with a fail-stop deadline |
| `gemm1` / `gemm2` (routed experts) | 128 × 512, 177–212 KB smem | 1 | up to 288 (48 rows × top-6, 8-row tiles) | No grid-wide wait; cluster launch control steals work |
| `markov` draft (cooperative) | 128 × 512 | sized from the GPU's SM count at runtime | — | Cooperative launch; checks co-residency itself |
| `wo_a_fused` | 38–82 × 256, cluster 8 | 2 | 256 | Cluster-local only |

- The two-shot kernel's worst grid does **not** fit co-resident even on GB300: 152 × 7 = 1,064 < 1,920. So its design cannot depend on 152 SMs; the B200 gives 1,036.
- The GEMM pair's persistent schedule uses CLC.
- The Lamport consumers' grids are at most 16 rows × cluster 5.
- `resident_blocks` is checked by the framework as `sms < resident_blocks → refuse`. Lowering it to 148 changes no kernel.

## 4. The patch

[`patches/sgl-nvidia-mega/0001`](../patches/sgl-nvidia-mega/0001-deepseek_v41-accept-B200-sm_100-148-SMs.patch) does two things:
- `isas={"sm_103", "sm_100"}`;
- `resident_blocks=148`.

[`0002`](../patches/sgl-nvidia-mega/0002-test-refusal-case-uses-sm_90.patch) changes one unit-test case. It asserted that `sm_100` is refused; the case now uses `sm_90`.

## 5. Offline validation, still without a GPU

1. **Framework:** SGLang PR #44945 applies cleanly on our production SGLang (`ed8c3139` plus the base recipe's patches). Its unit tests pass: 10/10, plus 12 subtests.
2. **Plugin:** `test/unit` passes 249/249 with the port.
3. **Collection:** all 685 kernel and e2e tests collect without import errors against our SGLang version.
4. **Launch line:** [`tools/check_launch.py`](../tools/check_launch.py) runs the framework's own launch guards on our exact command line, with the ISA set to `sm_100`. Result: **no refusal**.

## 6. On the GPU (production, before opening traffic)

| Gate | Result |
|---|---|
| `NVIDIA megakernel: active` on every TP rank | yes (target and draft specs) |
| NVIDIA's KL test of covered steps (`test_covered_steps_match_prefill`, 8 prompts × 256 tokens) | mean KL **0.0044**, bound 0.03 |
| GSM8K, first 500 | **491/500 = 0.982** |
| 16 functional checks + reasoning-leak check | 16/16, PASS |

**Not yet run:** the plugin's `test/kernels` suite (bitwise checks against SGLang's kernels, fail-stop) and the rest of `test/e2e`. They need the GPUs free, and we chose not to take production down for longer. They are next, and the results will be added here.

## 7. Not tuned yet
- **GEMM tactics.** `mxfp8_ll` pins its low-latency GEMM tactics for `(103, N, K)` only. On the B200 it takes FlashInfer's first tactic. Timing the offered tactics per (N, K, row bucket) on the B200 and pinning the best is the obvious next step.
- **Boot time.** The first boot took 18 minutes. We have not yet measured how much warm boots save.
