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

## 2. The megakernel window: results (2026-10-20 01:41 → 14:52 UTC)

The window ended early: the megakernel hung production twice and we moved back to the standard path at 14:52 UTC (§5). These numbers cover the whole megakernel run: 12.5 h of service, from night through the Brazilian business morning (22:41 → 11:52 BRT). Same tool and definitions as §3. Aggregates only: [`data/megakernel_window_20261020.json`](data/megakernel_window_20261020.json).

Per-user decode, tok/s, by requests running in the step. "All" counts every decode log line. "Clean" counts only lines with no prefill chunk in the same logging interval: the decode path alone. A hung server logs no decode lines, so the outages do not enter these tables; §5 reports them.

**A. Same clock hours.** The standard path's 01:41 → 14:52 UTC on 2026-10-19 against the megakernel's 01:41 → 14:52 UTC on 2026-10-20, with the maintenance gap removed from both. This controls for the time-of-day traffic mix. Computed with [`tools/same_clock_hours.py`](../tools/same_clock_hours.py), which reuses the protocol's parser unchanged.

| Running | Lines, before / after | All, mean: before → after | Clean, mean: before → after |
|---|---|---|---|
| 1 | 20,410 / 36,311 | 536 → **783 (+46%)** | 624 → **924 (+48%)** |
| 2–8 | 60,719 / 66,462 | 328 → **462 (+41%)** | 460 → **676 (+47%)** |
| 9–16 | 6,016 / 5,971 | 194 → 195 (+1%) | 293 → 323 (+10%) |
| 17–32 | 195 / 417 | 150 → 154 (+3%) | 291 → 281 (−3%) |

The same hours split finer (p50, the base repo's reference bands):

| Running | All, p50: before → after | Clean, p50: before → after |
|---|---|---|
| 1 | 492 → 767 (+56%) | 546 → 838 (+54%) |
| 2–4 | 363 → 531 (+46%) | 461 → 678 (+47%) |
| 5–8 | 236 → 266 (+13%) | 360 → 459 (+27%) |
| 9–16 | 182 → 185 (+1%) | 289 → 322 (+11%) |
| 17–32 | 147 → 153 (+4%) | 291 → 278 (−4%) |

**B. Against the whole 24 h before the swap** (the protocol's comparison; the megakernel side is 12.5 h):

| Running | Standard path (24 h) mean / p50 | Megakernel (12.5 h) mean / p50 | Clean mean, before → after |
|---|---|---|---|
| 1 | 538 / 492 | 783 / 767 | 633 → 924 |
| 2–8 | 321 / 294 | 462 / 411 | 456 → 676 |
| 9–16 | 189 / 177 | 195 / 185 | 294 → 323 |
| 17–32 | 144 / 141 | 154 / 153 | 260 → 281 |

How to read it:
- **1–4 requests running: +46% to +56% per user**, with and without prefill mixed in. These bands hold 79% of the decode lines in the megakernel window (71% in the same hours a day earlier).
- **The gain fades with load:** +13% (all) and +27% (clean) at 5–8, and **none at 9–16**, now on ~6k lines per side. This matches the plugin's design: its fused path covers decode steps of up to 48 rows, i.e. 8 requests at DSpark's 6 rows each. Above that, most of the step runs SGLang's regular kernels.
- **17–32: no gain either**, on few lines (195 / 417).
- **DSpark acceptance is unchanged** (mean accept length 3.85 → 3.78 at 1 running, 3.70 → 3.68 at 2–8, same hours).

Secondary, whole windows (not hour-matched; the megakernel side includes its first hour at memory 0.80):

| | Standard path (24 h) | Megakernel (12.5 h) |
|---|---|---|
| TTFT p50 / p90 / p99 | 0.51 / 1.81 / 9.42 s | 0.44 / 1.71 / 10.98 s |
| End-to-end latency p50 / p90 / p99 | 1.76 / 7.92 / 34.1 s | 1.46 / 6.64 / 29.7 s |
| Prefix-cache hit | 97.1% | 96.5% |
| Requests per hour | 6,022 | 6,152 |
| Chat completions | 144,138 × 200, 10 × 400 | 75,606 × 200, 4 × 400 |
| GPU memory free per GPU, minimum | 2.2 GiB | 4.4 GiB |
| Server hangs | 0 | **2** (§5) |

The first 16 minutes and the 8-hour partial, reported here earlier, are kept in [`data/early_window_20261020.json`](data/early_window_20261020.json) and [`data/partial_20261020_1013utc.json`](data/partial_20261020_1013utc.json). This section supersedes them.

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

**Deviation, recorded 2026-10-20 18:30 UTC, after the hangs:**
- The megakernel window ends at 14:52:14 UTC, when the second hang moved production to the standard path (§5). It covers 12.5 h of service instead of 24 h.
- The two outages (14:06–14:21 and 14:52–15:04 UTC) stay inside the window and count as failures of the megakernel run. Decode and latency statistics come only from the time the server was serving.
- We added a same-clock-hours comparison (§2 A) to the protocol's whole-day comparison (§2 B), because the shortened window covers only part of the day.

## 4. Final results (scheduled run, 2026-10-21 01:50 UTC)

The scheduled run recomputed the shortened window (§3) with the same tool, and it reproduces §2 exactly. Its summary step expected three windows (a leftover from the base repo's script) and stopped; the tables below come from the run's raw output, aggregated without client identifiers: [`data/results_24h_mega.json`](data/results_24h_mega.json).

**Per-user decode, tok/s**, the protocol's comparison: the megakernel's 01:41 → 14:52 UTC (12.5 h of service) against the standard path's 24 h before the swap. Before → after (change). The hour-matched comparison is §2 A.

| Running | Lines, before / after | All, mean | All, p50 | All, p90 | Clean, mean | Clean, p50 | Clean, p90 |
|---|---|---|---|---|---|---|---|
| 1 | 29,651 / 36,311 (clean 20,714 / 27,443) | 538 → 783 (+46%) | 492 → 767 (+56%) | 889 → 1,269 (+43%) | 633 → 924 (+46%) | 552 → 838 (+52%) | 965 → 1,326 (+37%) |
| 2–8 | 113,990 / 66,462 (clean 43,869 / 29,538) | 321 → 462 (+44%) | 294 → 411 (+40%) | 515 → 791 (+53%) | 456 → 676 (+48%) | 433 → 656 (+52%) | 606 → 900 (+49%) |
| 9–16 | 13,464 / 5,971 (clean 1,681 / 397) | 188 → 195 (+4%) | 177 → 185 (+4%) | 274 → 266 (−3%) | 294 → 323 (+10%) | 291 → 322 (+11%) | 340 → 371 (+9%) |
| 17–32 | 559 / 417 (clean 6 / 4) | 144 → 154 (+7%) | 141 → 153 (+8%) | 187 → 188 (+1%) | not measured (6 / 4 clean lines) | | |
| 33–64 | 5 / 0 | not measured | | | | | |

**Secondary**, whole windows:

| | Standard path (24 h) | Megakernel (12.5 h) |
|---|---|---|
| TTFT p50 / p90 / p99 | 0.51 / 1.81 / 9.42 s | 0.44 / 1.71 / 10.98 s |
| Queue time p50 / p90 / p99 | 0.06 / 0.72 / 8.82 s | 0.04 / 0.82 / 9.52 s |
| End-to-end latency p50 / p90 / p99 | 1.76 / 7.92 / 34.1 s | 1.46 / 6.64 / 29.7 s |
| Prefix-cache hit | 97.1% | 96.5% |
| DSpark mean accept length at 1 / 2–8 / 9–16 running | 3.86 / 3.70 / 3.63 | 3.78 / 3.68 / 3.69 |
| Prefill, full 4,096-token chunks, mean tok/s: all / with decode running / idle | 13,401 / 12,650 / 20,450 | 16,164 / 15,207 / 21,336 |
| HTTP 5xx | 4 | 7 |
| Aborted requests per hour | 15.2 | 11.6 |
| GPU memory free per GPU, p50 / minimum | 2.2 / 2.2 GiB | 7.6 / 4.4 GiB |
| Server hangs | 0 | **2** (§5) |

Caveats, as declared in §3 before the data:
- The two days carried different traffic. §2 A controls for the time of day; this table does not.
- The megakernel window starts with a cold prefix cache (restart at 01:20 UTC) and includes its first hour at memory 0.80.
- The megakernel runs MoE in TP4 (a plugin requirement); the standard path ran EP4. Differences at 9+ running and in prefill mix the megakernel with that layout change.
- The window is 12.5 h, not 24 h, and both outages count as failures of the megakernel run (§3 deviation).

Verdict:
- **1–8 requests running: +40% to +56% per user** (mean and p50, with and without prefill mixed in). These bands hold 94% of the megakernel window's decode lines.
- **9–16: a tie** (+4% here, +1% hour-matched; +10% in clean stretches). **17–32: no clear change**, on few lines. 33–64: not measured.
- **Worse:** two hangs in 12.5 h, which moved production back to the standard path; TTFT p99 10.98 vs 9.42 s and queue time p90/p99 slightly higher, on different traffic hours; 7 vs 4 HTTP 5xx.
- **Not production-ready** until the cause of the hangs is fixed and confirmed (§5).

## 5. Two hangs on 2026-10-20

| | Hang 1 | Hang 2 |
|---|---|---|
| Scheduler's last logged batch | 14:06:11 UTC | 14:52:14 UTC |
| Time since the server started | ~11 h | 31 min |
| Load at that moment | 6 running, queue 0 | 14 running, queue 0 |
| Batch in flight (SGLang's watchdog dump) | 1 request, still in prefill (no output yet) | 1 request |
| GPU power during the hang (5-minute samples) | GPUs 0, 1, 3 at ~670 W and 100% busy; GPU 2 at ~205 W | all four at ~200 W |
| Per-rank KV accounting in the dump | rank 2 differs from ranks 0, 1 and 3 by 1,024 tokens, in both the full and the SWA pool | ranks 0 and 3 differ from ranks 1 and 2 by 768 tokens |
| Recovery | SGLang's watchdog ended the server at 14:13; automatic restart, serving again at 14:21 | SGLang's watchdog ended it at ~14:59; our watchdog's circuit breaker restarted on the standard path, serving at 15:04 |
| Out of service | ~15 min | ~12 min |

What this shows:
- The four TP ranks of one server must hold identical scheduler state: they run in lockstep on the same inputs. At both hangs they did not, and a collective waited for ranks that were doing something else.
- The megakernel's own cross-GPU waits have a 60-s fail-stop deadline, and it never fired: the stalled collective was outside the megakernel's kernels.
- No hardware fault: no ECC errors, no pending row remaps and no hardware slowdown on any GPU.
- Load was moderate both times.

What we are doing:
- Production stays on the standard path until a fix is confirmed.
- A fix is being tested, together with a per-rank divergence probe and a watchdog that falls back to the standard path before a hang. We will publish the cause and the fix once they are confirmed.
