#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 on 4x B200 with NVIDIA's decode megakernel ("SGLang low-latency mode"): SGLang + the
# megakernel framework (sgl-project/sglang#44945) + sgl-nvidia-mega ported to the B200 (see ../patches). In production
# on our machine since 2026-10-20 01:40 UTC.
#
# Same recipe as https://github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200 (Engram in host RAM per rank,
# mem 0.80, chunk 4096, 16 decode rounds between prefill chunks, at most 64 running, DSpark block 5, thinking on at
# effort high) plus what the plugin requires: --ep-size 1 (MoE in TP4, every expert local), --dsv4-attn-backend
# flashmla, --kv-cache-dtype fp8_e4m3, SGLANG_SPEC_TP_SYNC=rng,-dspark-accept-sample, --enable-nvidia-megakernel.
#
#   API_KEY=... [ADMIN_API_KEY=...] MODEL_PATH=/models/dsv41-1017 \
#   [IMG=/path/to/extracted/image] [SGLANG_OVERLAY=/overlay/site:/patched/sglang/python] bash serve/serve_megakernel.sh
#   DRY=1 bash serve/serve_megakernel.sh   prints the arguments (for tools/check_launch.py)
# Memory 0.84, not the base recipe's 0.80: TP4 MoE weights take ~25 GB/GPU more, and at 0.80 the KV pool is 1.85 M tokens.
# Set ADMIN_API_KEY: without it, SGLang lets the ordinary key call admin routes (sgl-project/sglang#43706).
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/models/dsv41-1017}"
ARGS=(
  --model-path "$MODEL_PATH" --served-model-name "${SERVED_MODEL_NAME:-deepseek-v4.1-flash}" --trust-remote-code
  --tp-size 4 --ep-size 1 --mem-fraction-static "${MEMF:-0.84}"
  --chunked-prefill-size 4096 --prefill-decode-interval "${PDI:-16}" --swa-prefix-tails 4096
  --max-running-requests 64 --cuda-graph-max-bs-decode 64
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 --schedule-policy lpm
  --enable-multimodal --image-processor-backend pil --enable-cache-report --enable-metrics --media-url-max-file-size-mb 1024
  --kv-cache-dtype fp8_e4m3 --dsv4-attn-backend flashmla --enable-nvidia-megakernel
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
)
if [ "${DRY:-0}" = 1 ]; then printf '%s\n' "${ARGS[@]}"; exit 0; fi
: "${API_KEY:?set API_KEY}"
ARGS+=(--api-key "$API_KEY"); [ -n "${ADMIN_API_KEY:-}" ] && ARGS+=(--admin-api-key "$ADMIN_API_KEY")
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_SPEC_TP_SYNC="rng,-dspark-accept-sample"
export NCCL_CUMEM_ENABLE=1 SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_DEFAULT_THINKING="${THINKING:-1}" SGLANG_DSV41_REASONING_EFFORT="${EFFORT:-high}"
[ -n "${SGLANG_OVERLAY:-}" ] && export PYTHONPATH="$SGLANG_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${IMG:-}" ]; then PY="$IMG/opt/sglang/bin/python"; export PATH="$IMG/opt/sglang/bin:$PATH"; else PY=python3; fi
RUN_DIR="${RUN_DIR:-/tmp/sglang-mega-run}"; mkdir -p "$RUN_DIR"; cd "$RUN_DIR"   # empty dir: nothing shadows the overlay
exec "$PY" -m sglang.launch_server "${ARGS[@]}"
