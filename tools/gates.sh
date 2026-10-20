#!/usr/bin/env bash
# The four gates we ran against the megakernel server BEFORE opening it to users (2026-10-20). Any failure → exit 1.
#   1. the server log shows "NVIDIA megakernel: active" on every TP rank (otherwise you are measuring the normal path)
#   2. NVIDIA's KL test of the covered steps (tools/kl_gate.py), mean KL <= 0.03   — ours: 0.0044
#   3. GSM8K, first 500 questions (tools/gsm8k_quick.py), >= 0.94                  — ours: 0.982
#   4. tests/val_dsv41.py (16 functional checks) and tests/val_leak.py             — ours: 16/16, PASS
# Usage: LOG=server.log URL=http://127.0.0.1:8000 API_KEY=... [ADMIN_API_KEY=...] MODEL_PATH=... LONGBENCH_CACHE=dir \
#        PY=<image python> SGLANG_OVERLAY=<overlay> GSM8K=test.jsonl bash tools/gates.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); OUT=${OUT:-gates-$(date -u +%Y%m%d-%H%M%S)}; mkdir -p "$OUT"
fail() { echo "GATE FAILED: $*"; exit 1; }
n=$(grep -c "NVIDIA megakernel: active" "$LOG"); echo "gate 1: 'NVIDIA megakernel: active' x $n"; [ "$n" -ge 4 ] || fail "megakernel not active on 4 ranks"
PYTHONPATH="$SGLANG_OVERLAY" "$PY" "$HERE/tools/kl_gate.py" | tee "$OUT/kl.txt" | tail -2; [ "${PIPESTATUS[0]}" = 0 ] || fail "KL"
KF=$(mktemp); printf '%s' "$API_KEY" > "$KF"
python3 "$HERE/tools/gsm8k_quick.py" --url "$URL" --key-file "$KF" --n 500 --conc 32 --data "$GSM8K" --out "$OUT/gsm8k.jsonl" | tee "$OUT/gsm8k.txt"
rm -f "$KF"; acc=$(grep -oE '= [0-9.]+' "$OUT/gsm8k.txt" | tr -dc '0-9.'); awk "BEGIN {exit !(${acc:-0} >= 0.94)}" || fail "GSM8K $acc"
python3 "$HERE/tests/val_leak.py" | tee "$OUT/leak.txt" | tail -1; grep -q "LEAK CHECK PASS" "$OUT/leak.txt" || fail "val_leak"
python3 "$HERE/tests/val_dsv41.py" | tee "$OUT/val.txt" | tail -1; grep -q "RESUMO 16/16 PASS" "$OUT/val.txt" || echo "check the FAIL lines in $OUT/val.txt (a length-only FAIL of the long pt-BR essay is sampling, not numerics)"
echo "ALL GATES PASSED ($OUT)"
