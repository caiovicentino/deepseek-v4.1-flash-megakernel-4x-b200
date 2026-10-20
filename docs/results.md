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

## 2. Partial results while the 24-hour window runs

**Latest update: 2026-10-20 10:13 UTC, 8.0 h into the megakernel window** (01:41 → 10:13 UTC, minus the 23-minute maintenance gap in §3). So far the window covers only night and early morning in Brazil (22:41 → 07:13 BRT). Same tool and definitions as §3. Aggregates only: [`data/partial_20261020_1013utc.json`](data/partial_20261020_1013utc.json).

Per-user decode, tok/s, by requests running in the step. "All" counts every decode log line. "Clean" counts only lines with no prefill chunk in the same logging interval: the decode path alone.

**A. Same clock hours.** The standard path's 01:41 → 10:13 UTC on 2026-10-19, with the same gap removed, against the megakernel's 01:41 → 10:13 UTC on 2026-10-20. This controls for the time-of-day traffic mix, which the protocol's whole-day comparison (B) does not. Computed with [`tools/same_clock_hours.py`](../tools/same_clock_hours.py), which reuses the protocol's parser unchanged.

| Running | Lines, before / after | All, mean: before → after | Clean, mean: before → after |
|---|---|---|---|
| 1 | 17,963 / 29,217 | 535 → **780 (+46%)** | 620 → **919 (+48%)** |
| 2–8 | 39,911 / 46,823 | 339 → **477 (+41%)** | 466 → **684 (+47%)** |
| 9–16 | 1,515 / 1,258 | 204 → 197 (−4%) | 284 → 331 (+17%; 360 / 86 clean lines) |
| 17–32 | 18 / 0 | not reached | not reached |

The same hours split finer (p50, the base repo's reference bands):

| Running | All, p50: before → after | Clean, p50: before → after |
|---|---|---|
| 1 | 492 → 760 (+54%) | 542 → 828 (+53%) |
| 2–4 | 367 → 536 (+46%) | 462 → 677 (+47%) |
| 5–8 | 237 → 264 (+11%) | 355 → 460 (+29%) |
| 9–16 | 197 → 187 (−5%) | 281 → 323 (+15%) |

**B. Against the whole 24 h before the swap** (the protocol's comparison; the megakernel side is still partial):

| Running | Standard path (24 h) mean / p50 | Megakernel (8.0 h) mean / p50 | Clean mean, before → after |
|---|---|---|---|
| 1 | 538 / 492 | 780 / 760 | 633 → 919 |
| 2–8 | 321 / 294 | 477 / 445 | 456 → 684 |
| 9–16 | 189 / 177 | 197 / 187 | 294 → 331 |
| 17–32 | 144 / 141 (559 lines) | not reached | — |

How to read it:
- **1–4 requests running: ~+45–55% per user**, in both comparisons, with and without prefill mixed in. Tonight these bands hold 87% of decode lines.
- **The gain shrinks as more requests share a step:** +11% (all) and +29% (clean) at 5–8. That fits the megakernel's design target, low-batch latency. The 2–8 mean (+41%) is helped by a lighter mix inside the band: 82% of its lines are at 2–4 after, 76% before.
- **9–16: inconclusive.** All lines and clean lines disagree in sign, with ~1.3k lines and only 86 clean lines after. 17+ was not reached. Daytime traffic should fill these bands.
- **DSpark acceptance is unchanged** (mean accept length 3.84 → 3.76 at 1 running, 3.69 → 3.65 at 2–8, same hours). The speedup comes from faster steps, not from more tokens per step.

Secondary, whole windows (not hour-matched, so the traffic differs; the megakernel side includes the first hour at memory 0.80):

| | Standard path (24 h) | Megakernel (8.0 h) |
|---|---|---|
| TTFT p50 / p90 / p99 | 0.51 / 1.81 / 9.42 s | 0.39 / 1.08 / 7.47 s |
| End-to-end latency p50 / p90 / p99 | 1.76 / 7.92 / 34.1 s | 1.16 / 5.22 / 19.2 s |
| Prefix-cache hit | 97.1% | 96.7% |
| Requests per hour | 6,022 | 5,558 |
| Chat completions | 144,138 × 200, 10 × 400 | 45,484 × 200, 4 × 400 |
| OOM / tracebacks / retractions / failed health checks | 0 / 2 / 0 / 3 | **0 / 0 / 0 / 0** |
| GPU memory free per GPU, minimum | 2.2 GiB | 7.6 GiB |

KV pool: it peaked at 90% in the first hour, at memory 0.80 (1.85 M tokens). Since the move to 0.84 (6.39 M tokens), the highest usage in the decode log lines is 25%.

The first 16 minutes, reported here earlier as a preliminary table, are kept in [`data/early_window_20261020.json`](data/early_window_20261020.json). This section supersedes them. The 24-hour result in §4 will supersede this section.

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
