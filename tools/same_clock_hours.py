"""Same-clock-hours comparison for the partial updates in docs/results.md §2.

Runs the base repo's measure24h log parser (unchanged) over two windows of equal clock hours: the megakernel window
from its start to CUT, and the same UTC hours one day earlier on the standard path, with the megakernel window's
declared gaps shifted by one day and removed from both. This controls for the time-of-day traffic mix while the
24-hour window is still running. Output: per-band decode aggregates only (no clients, paths or events).

Usage: python3 same_clock_hours.py MEASURE24H_DIR "YYYY-MM-DD HH:MM:SS" OUT.json
  MEASURE24H_DIR holds the measure24h script whose WINDOWS table has the "dsv41_mega" and "dsv41_pre" windows
  (docs/results.md §3); it is imported as the module `measure24h_mega`.
"""
import json
import sys
from datetime import datetime, timedelta

sys.path.insert(0, sys.argv[1])
import measure24h_mega as m  # noqa: E402

FMT = "%Y-%m-%d %H:%M:%S"
cut = sys.argv[2]
day = timedelta(days=1)


def shift(s):
    return (datetime.strptime(s, FMT) - day).strftime(FMT)


mega, pre = dict(m.WINDOWS["dsv41_mega"]), dict(m.WINDOWS["dsv41_pre"])
mega["t1"] = cut
pre["t0"], pre["t1"] = shift(mega["t0"]), shift(cut)
pre["gaps"] = [[shift(a), shift(b), "same clock interval as the megakernel window's gap"] for a, b, _ in mega["gaps"]]
m.WINDOWS = {"mega_same_hours": mega, "pre_same_hours": pre}
r = m.scan_logs()

out = {"cut_utc": cut,
       "windows": {k: {"t0": w["t0"], "t1": w["t1"], "gaps": w["gaps"]} for k, w in m.WINDOWS.items()},
       "decode": {}, "decode_reference_bands": {}, "running_req": {}}
for k, v in r.items():
    out["decode"][k] = {b: {"lines": x.get("lines"), "lines_clean": x.get("lines_clean"),
                            "per_user_all": {q: (x.get("per_user_all") or {}).get(q) for q in ("n", "mean", "p50")},
                            "per_user_clean": {q: (x.get("per_user_clean") or {}).get(q) for q in ("n", "mean", "p50")},
                            "accept_len_mean": (x.get("accept_len") or {}).get("mean"),
                            "share_of_decode_lines": x.get("share_of_decode_lines")}
                        for b, x in v["decode"].items()}
    out["decode_reference_bands"][k] = v["decode_reference_bands"]
    out["running_req"][k] = v["running_req_over_decode_lines"]
json.dump(out, open(sys.argv[3], "w"), indent=1)


def change(a, b):
    return f"{(b / a - 1) * 100:+.0f}%" if a and b else "-"


for b in ("1", "2-8", "9-16", "17-32"):
    p, q = out["decode"]["pre_same_hours"][b], out["decode"]["mega_same_hours"][b]
    pa, qa = p["per_user_all"]["mean"], q["per_user_all"]["mean"]
    pc, qc = p["per_user_clean"]["mean"], q["per_user_clean"]["mean"]
    print(f"{b:>5}  lines {p['lines']}/{q['lines']}  all {pa} -> {qa} {change(pa, qa)}  clean {pc} -> {qc} {change(pc, qc)}")
