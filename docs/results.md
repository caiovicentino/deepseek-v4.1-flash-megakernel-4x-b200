# Results

## 1. Before opening traffic (2026-10-20 01:39–01:40 UTC, production port, megakernel on, no users)

Measured with [`tools/bench_engines.py`](../tools/bench_engines.py) on token-id prompts from [`tools/make_prompts.py`](../tools/make_prompts.py): 1,024 output tokens with `ignore_eos`, temperature 0.6, top_p 0.95. Raw rows are in [`data/preopen_20261020.json`](data/preopen_20261020.json).

| Cell | Per-user decode (median) | TTFT p50 |
|---|---|---|
| 8k context, 1 user | **829 tok/s** | 0.15 s |
| 8k context, 4 users | **465 tok/s** | 0.20 s |
| LocalMaxxing-style: coding prompt ~2.85k tokens in, 2,048 out, batch 1 | **1,136 tok/s** | 0.14 s |

For reference:
- the community's fastest LocalMaxxing run for this checkpoint is 502.7 tok/s on 4× RTX PRO 6000 Max-Q (unverified, WIP megakernel on vLLM);
- NVIDIA/LMSYS report 928 tok/s P90 per user at one AgentX session on 4× GB300.

The workloads differ, so read these as order-of-magnitude comparisons.

Quality gates at the same moment:
- KL of the covered steps 0.0044 (bound 0.03);
- GSM8K 491/500;
- 16/16 functional checks;
- reasoning-leak PASS.

## 2. First minutes of real traffic (preliminary)

These come from the 24-hour tool (§3), run while the megakernel window had only its first ~16 minutes ([`data/early_window_20261020.json`](data/early_window_20261020.json)). Per-user decode tok/s, by running requests, from the server's decode log lines:

| Running | Standard path (24 h before) mean / p50 | Megakernel (first 16 min) mean / p50 | Clean lines* mean, before → after |
|---|---|---|---|
| 1 | 538 / 492 | 731 / 705 | 633 → 1,018 |
| 2–8 | 321 / 294 | 398 / 327 | 456 → 657 |
| 9–16 | 189 / 177 | 173 / 171 (n = 36) | too few |

\* "Clean" = decode lines with no prefill chunk in the same logging interval: the decode path alone.

Sixteen minutes is not a measurement. The 24-hour result replaces this table.

## 3. 24-hour measurement: protocol (written 2026-10-20, before the data)

| | |
|---|---|
| **Windows** | Megakernel: 2026-10-20 01:41 → 2026-10-21 01:41 UTC. Standard path: 2026-10-19 00:59 → 2026-10-20 00:59 UTC, the last full 24 h before the swap, same weekday pattern |
| **Tool** | The base repo's [`tools/measure24h`](https://github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200/tree/main/tools/measure24h), unchanged. Only its window table is replaced (two windows above, log files and server pids resolved at run time). Read-only over the server logs and the 5-minute `/metrics` + `nvidia-smi` series. No request content enters the output |
| **Primary** | Per-user decode by running-request band (1, 2–8, 9–16, 17–32, 33–64): mean, p50, p90, all lines and clean lines |
| **Secondary** | TTFT p50/p90/p99 from the histogram deltas; prefix-cache hit; DSpark acceptance; prefill throughput; 5xx; GPU memory over time |
| **Caveats declared now** | The two days carry different traffic. The megakernel window starts with a cold prefix cache (restart at 01:20). Bands that traffic does not reach stay "not measured". The megakernel run uses MoE in TP4 (plugin requirement) while the standard path used EP4 |

**Deviation, recorded 2026-10-20 03:10 UTC, before any 24-hour result:**
- The megakernel window has a gap from 02:40:49 to 03:03:38 UTC: a maintenance that ran the plugin's kernel tests and raised `--mem-fraction-static` from 0.80 to 0.84.
- The memory change grows the KV pool from 1.85 M to 6.39 M tokens. It does not change the decode kernels.
- Decode per user is compared over the whole window. Cache hit rate and TTFT are also reported for the part after 03:03:38 alone.

## 4. 24-hour results

*Pending: computed automatically at 2026-10-21 01:50 UTC and added here.*
