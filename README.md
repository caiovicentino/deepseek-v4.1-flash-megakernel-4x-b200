# DeepSeek-V4.1-Flash-1017 on 4× NVIDIA B200 with NVIDIA's decode megakernel

This recipe runs NVIDIA's decode megakernel for **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** on **four B200**. The megakernel is
[`sgl-nvidia-mega`](https://github.com/sgl-project/sgl-nvidia-mega), the
["SGLang low-latency mode"](https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode). NVIDIA ships it only for GB300
(SM103). We ported it to the B200 (SM100) and put it in production on the machine that serves our community.

> **Status: 2026-10-20 (live, updated as the measurement runs)**
> - **In production since 2026-10-20 01:40 UTC** on our 4× B200. It replaced the standard SGLang path of the
>   [base recipe](https://github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200). Same model, same image, same
>   patches; only the flags the plugin needs and the plugin itself changed.
> - **Gates passed before any user reached it.** All four ran against the production port, with traffic closed:
>   - the megakernel is active on every TP rank;
>   - NVIDIA's own KL test of the covered steps gives a **mean KL of 0.0044** (bound 0.03);
>   - **GSM8K 491/500 = 98.2%**;
>   - 16/16 functional checks (tool calls, images, streaming, long pt-BR generation) and the reasoning-leak check pass.
> - **Speed measured right after boot, before opening traffic** (per-user decode, median):
>   - **829 tok/s** with 1 user at 8k context, and **465 tok/s** per user with 4 users;
>   - **1,136 tok/s** on a LocalMaxxing-style single stream (coding prompt, ~2.85k tokens in, 2,048 out, batch 1).
> - **Partial result, 8.0 h into the 24-hour window** (2026-10-20 10:13 UTC). So far this covers only night and early
>   morning in Brazil. Same tool and definitions as the 24-hour measurement, against the same clock hours one day earlier
>   on the standard path. Per-user decode:
>   - 1 request running: **780 vs 535 tok/s (+46%)**; 919 vs 620 (+48%) in stretches with no prefill mixed in.
>   - 2–8 running: **477 vs 339 (+41%)**; 684 vs 466 (+47%) clean. The gain is largest at 2–4 (+46%) and smaller at 5–8
>     (+11% all lines, +29% clean).
>   - 9–16 running: inconclusive so far (−4% all lines, +17% clean, few samples). 17+ not reached yet.
>   - DSpark acceptance unchanged. 0 OOM, 0 tracebacks, 0 failed health checks.
>   - Tables, the comparison against the whole previous day, TTFT and memory: [`docs/results.md`](docs/results.md) §2.
> - **Update 2026-10-20 03:03 UTC: memory, kernel tests and boot time.**
>   - **Memory:** the megakernel needs MoE in TP4 (`--ep-size 1`), and that costs ~25 GB more weights per GPU. With
>     the SWA window fixed at ~37 GB, the main KV pool fell from 20 M to **1.85 M tokens** at `--mem-fraction-static
>     0.80`. We moved to **0.84**: pool **6.39 M tokens**, ~12 GB/GPU still free after boot.
>   - **Kernel tests on the B200:** the plugin's own suite now ran on our GPUs (`test/kernels`). It passed:
>     - 592 single-GPU tests (2 failures, both explained in [`docs/porting-to-b200.md`](docs/porting-to-b200.md) §6);
>     - the 4-GPU hops (2/2), hop B (7/7) and fail-stop (5/5).
>   - **Gates again:** KL 0.0045, GSM8K 0.974, 16/16, leak PASS. Speed unchanged (803 / 480 / 1,141 tok/s).
>   - **Boot time:** a warm boot takes **6 min 16 s**. The first boot took 18 min because the plugin's cache directory
>     started empty.
> - **24-hour measurement running** (2026-10-20 01:41 → 2026-10-21 01:41 UTC). It uses the same tool and definitions as the base repo and compares against the 24 h before the swap. Results land in [`docs/results.md`](docs/results.md) around 2026-10-21 02:00 UTC.

Series:
- [GLM-5.3-Flash 4× RTX PRO 6000](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-rtx-pro-6000)
- [GLM-5.3-Flash 2× B200](https://github.com/caiovicentino/glm-5.3-flash-sglang-2x-b200)
- [GLM-5.3-Flash 4× B200](https://github.com/caiovicentino/glm-5.3-flash-sglang-4x-b200)
- [DeepSeek-V4.1-Flash 4× B200 (base recipe)](https://github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200)
- this repo.

## Why a megakernel helps here

With one user, a DSpark decode step on the standard path took **~6.1 ms** on our B200. The memory-bandwidth floor for that step is **~1 ms**: the weights the step actually touches, read once.

The difference is kernel-launch latency, host work and small all-reduces between many small kernels. NVIDIA counted ~1,600 kernels per step on stock vLLM on GB300; we did not count ours. That gap is what NVIDIA's specialized decode attacks:
- the attention output projection writes straight into an NVLink-multicast Lamport all-reduce fused with the mHC post;
- routed experts run as 8-row-tile block-scaled `tcgen05` GEMMs, with routing done by bitmap;
- the Engram lookups are prefetched on a side stream;
- the five-step DSpark draft runs in one kernel per GPU;
- programmatic dependent launch (PDL) chains everything.

On GB300, LMSYS reports P90 interactivity going from **671 to 928 tok/s per user** at one session.

## Quick start

Start from the [base recipe](https://github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200): its image and its two patches.

```bash
git clone https://github.com/caiovicentino/deepseek-v4.1-flash-megakernel-4x-b200 && cd deepseek-v4.1-flash-megakernel-4x-b200
git clone https://github.com/sgl-project/sgl-nvidia-mega
# 1) the megakernel framework into SGLang (PR #44945) + our B200 port of the plugin
bash patches/apply.sh /sgl-workspace/sglang ./sgl-nvidia-mega
# 2) install the plugin where only the megakernel server sees it (production keeps running untouched)
pip install --no-deps --target /opt/mega-site ./sgl-nvidia-mega
export SGLANG_OVERLAY=/opt/mega-site:/sgl-workspace/sglang/python
# 3) offline checks, no GPU needed
python -I tools/check_nvrtc_sm100.py ./sgl-nvidia-mega          # 34/34 valid kernel variants compile for sm_100a
DRY=1 bash serve/serve_megakernel.sh > args.txt
ARGS_FILE=args.txt CUDA_VISIBLE_DEVICES=9 PYTHONPATH=$SGLANG_OVERLAY python tools/check_launch.py   # "NO REFUSAL"
# 4) serve, then run the gates BEFORE sending users to it
API_KEY=... ADMIN_API_KEY=... MODEL_PATH=/models/dsv41-1017 bash serve/serve_megakernel.sh > server.log 2>&1 &
LOG=server.log URL=http://127.0.0.1:8000 API_KEY=... ADMIN_API_KEY=... MODEL_PATH=/models/dsv41-1017 \
  LONGBENCH_CACHE=/tmp/longbench PY=python3 GSM8K=test.jsonl bash tools/gates.sh
```

**Boot time.** The first boot took **18 minutes**, almost all of it filling an empty cache directory:
- 4 min of JIT for the mHC prewarm;
- ~6 min of FlashInfer autotuning.

With the cache warm, a boot takes **6 min 16 s**: weights, Engram host tables (~2 min), the plugin's build (8 s for the target, 1.2 s for the draft) and CUDA-graph capture of 27 batch sizes (~3 min). Keep `SGLANG_CACHE_DIR` stable between boots.

## Recipe card

| | |
|---|---|
| hardware | 4× B200 (148 SMs and 179 GiB each, NVLink NV18, NVLink multicast supported), 2 TB host RAM; driver 595.91 |
| image | `lmsysorg/sglang:nightly-dev-cu13-20261016-a93f16f0`: SGLang source tree `ed8c3139`, torch 2.14.1+cu130, FlashInfer 0.7.1, nvidia-cutlass-dsl 4.8.0, cuda-python 13.4.4. Plus the base recipe's two patches |
| framework | SGLang PR [#44945](https://github.com/sgl-project/sglang/pull/44945) (head `5ba9bb2dc2`), its python/ part and unit tests. It applies cleanly on `ed8c3139`; [`patches/sglang-pr44945-applied.diff`](patches/sglang-pr44945-applied.diff) |
| plugin | [`sgl-nvidia-mega`](https://github.com/sgl-project/sgl-nvidia-mega) `d47deb3` plus our 2-commit B200 port ([`patches/sgl-nvidia-mega/`](patches/sgl-nvidia-mega/)) |
| changes vs. the base recipe | `--ep-size 1` (MoE in TP4; the plugin needs every expert local), `--dsv4-attn-backend flashmla` (the base auto-resolves `flashmla_mega`), `--kv-cache-dtype fp8_e4m3`, `SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample`, `--enable-nvidia-megakernel` |
| memory | `--mem-fraction-static 0.84` (the base recipe uses 0.80): TP4 MoE weights take ~100 GB per GPU instead of 74.5, and at 0.80 the main KV pool would be 1.85 M tokens; at 0.84 it is 6.39 M |
| unchanged | TP4, Engram host table per rank, chunk 4096, 16 decode rounds between prefill chunks, max 64 running, DSpark block 5, thinking on at effort high, 1M context |
| covered steps | DSpark target-verify steps of 1–384 rows: up to 64 requests at 6 rows each. Prefill, extend and anything else runs SGLang's normal path |

## The port in one paragraph

The plugin compiles its CUDA kernels with NVRTC for whatever GPU it finds (`sm_{major}{minor}a`).

All 34 valid kernel variants compile for `sm_100a`; none uses an SM103-only instruction. FlashInfer's low-latency block-scaled GEMM, which the plugin calls, supports SM100 officially. The plugin's own kernel tests are written for any "SM100-family" GPU.

What held the B200 back was the spec: `isas={"sm_103"}` and `resident_blocks=152`. The second is the framework's guard against partitioned GPUs, not a kernel constant. The cross-GPU two-shot kernel does not fit co-resident even on GB300 (7 blocks per SM at 72 registers: 1,064 blocks against a grid of up to 1,920), and the expert GEMMs schedule with cluster launch control.

So the port is two lines: accept `sm_100` with 148 SMs. The details, the occupancy table and what we did NOT test are in [`docs/porting-to-b200.md`](docs/porting-to-b200.md).

## Operations

How we swapped production with a few minutes of downtime and an automatic way back is in [`ops/README.md`](ops/README.md):
1. a launcher override;
2. the gates;
3. the tunnel reopened only after the gates passed and only once it answered from outside.

To go back to the standard path, remove the override and restart: about 5 minutes.

**Security.** Set `--admin-api-key`. Without it, SGLang lets the ordinary API key call admin routes such as `/start_profile`, `/release_memory_occupation` and `/update_weights_from_disk` ([sglang#43706](https://github.com/sgl-project/sglang/issues/43706)). Also allow only `/v1/*` (plus whatever your clients use) at your edge: `/get_server_info` returns the API key in clear.

## Limits, stated plainly
- **Not an official NVIDIA configuration.** NVIDIA validates `sgl-nvidia-mega` on GB300 only. On the B200, our evidence is:
  - the offline checks above;
  - NVIDIA's KL test of the covered steps on the real model;
  - GSM8K and our functional checks, all on the production server before opening.
- **Kernel tests on our GPUs: done.** The plugin's suite ran on 2026-10-20, with 2 explained failures out of 594 single-GPU tests; all 4-GPU tests passed. See [`docs/porting-to-b200.md`](docs/porting-to-b200.md) §6.
- **GEMM tactics not tuned for the B200.** The plugin pins tactics for SM103 only. On SM100, FlashInfer's autotuner tunes the calls that go through it at boot, while the plugin's direct launches (`wqkv_a`, `wq_b`, the indexer projection) use FlashInfer's first tactic.
  - Our first tuning attempt timed a Python launch loop: ~45 µs per call for every shape, i.e. host overhead. We discarded it before it served a request.
  - A valid attempt must time launches inside a CUDA graph. We expect a small gain (a few %), if any.
- **KV capacity.** TP4 MoE costs ~25 GB per GPU of weights. Even at 0.84 the main KV pool (6.39 M tokens) is a third of the base recipe's (20 M), so prefix-cache churn may rise under heavy load. The 24-hour measurement reports the cache hit rate.
- **MoE in TP4.** The plugin requires it (`--ep-size 1`), while the base recipe runs EP4. Bands with many concurrent requests may behave differently. The 24-hour measurement covers whatever bands the traffic reaches.
- **Experimental.** The framework PR is still open, and NVIDIA marks the plugin experimental. Pin the commits listed above.

## Credits and license
- NVIDIA's `sgl-nvidia-mega` and the SGLang megakernel framework (PR #44945): Apache-2.0, see [`NOTICE.md`](NOTICE.md). The files in `patches/` are derived from them and keep that license.
- Everything else in this repo: MIT, CulturaBuilder.
