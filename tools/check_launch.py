"""Runs the megakernel framework's own launch checks (guards.megakernel_disable_reasons + plugin_reasons) on a
command line, WITHOUT a GPU, as if on a B200 (ISA sm_100). Lists every reason the plugin would refuse the launch.
Usage: DRY=1 bash serve/serve_megakernel.sh > args.txt
       ARGS_FILE=args.txt CUDA_VISIBLE_DEVICES=9 PYTHONPATH=<overlay> <image python> tools/check_launch.py
(CUDA_VISIBLE_DEVICES=9 hides the GPUs without making the variable empty, which SGLang's test utils reject.)
"""
import os
import sys

ENV = {
    "SGLANG_SPEC_TP_SYNC": "rng,-dspark-accept-sample",
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
    "NCCL_CUMEM_ENABLE": "1",
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV41_REASONING_EFFORT": "high",
}
os.environ.update(ENV)
ARGS_PADRAO = [
    "--model-path", os.environ.get("MODEL_PATH", "/models/dsv41-1017"), "--served-model-name", "glm-5.3-flash", "--trust-remote-code",
    "--tp-size", "4", "--ep-size", "1",
    "--mem-fraction-static", "0.80", "--chunked-prefill-size", "4096", "--prefill-decode-interval", "16",
    "--swa-prefix-tails", "4096", "--max-running-requests", "64", "--cuda-graph-max-bs-decode", "64",
    "--speculative-algorithm", "DSPARK", "--speculative-dspark-block-size", "5",
    "--reasoning-parser", "deepseek-v41", "--tool-call-parser", "deepseekv41", "--schedule-policy", "lpm",
    "--enable-multimodal", "--image-processor-backend", "pil", "--enable-cache-report", "--enable-metrics",
    "--kv-cache-dtype", "fp8_e4m3", "--dsv4-attn-backend", "flashmla",
    "--enable-nvidia-megakernel", "--host", "127.0.0.1", "--port", "8100",
]

# With ARGS_FILE (output of `DRY=1 bash serve/serve_megakernel.sh`) it checks EXACTLY what will be launched.
ARGS = [a for a in open(os.environ["ARGS_FILE"]).read().split("\n") if a] if os.environ.get("ARGS_FILE") else ARGS_PADRAO
print("arguments:", "from " + os.environ["ARGS_FILE"] if os.environ.get("ARGS_FILE") else "built-in list", f"({len(ARGS)} items)")
import sglang.srt.megakernels.guards as g  # noqa: E402

g.device_isa = lambda: "sm_100"  # a B200, without touching a GPU
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.utils import get_exception_traceback  # noqa: E402,F401

import sgl_nvidia_mega.plugin as _plugin  # noqa: E402

_plugin.register()  # what SGLang's general-plugin loader does at boot
print("plugin:", g.plugin_reasons() or "instalado e registrado")
parser = __import__("argparse").ArgumentParser()
ServerArgs.add_cli_args(parser)
ns = parser.parse_args(ARGS)
try:
    sa = ServerArgs.from_cli_args(ns)
except Exception as e:  # noqa: BLE001
    # full resolution may need a GPU (memory-based defaults)
    print("full ServerArgs did not resolve without a GPU:", type(e).__name__, str(e)[:300])
    sa = None
from sglang.srt.configs.model_config import ModelConfig  # noqa: E402

cfg = g.resolving_view(sa) if sa is not None else None
if cfg is None:
    sys.exit("no resolved cfg: run the check on a test boot with the GPU")
if getattr(cfg, "cuda_graph_config", None) is None:
    # resolved only at boot (with a GPU); use what our production resolved: decode "full", up to 64
    from types import SimpleNamespace

    resolved = SimpleNamespace(decode=SimpleNamespace(backend="full", max_bs=64), prefill=SimpleNamespace(backend="disabled"))
    class _Com:  # delegates to the resolved cfg, except cuda_graph_config, which only exists at boot
        def __init__(self, base):
            self._base = base

        def __getattr__(self, k):
            return resolved if k == "cuda_graph_config" else getattr(self._base, k)

    cfg = _Com(cfg)
    print("cuda_graph_config: injected the production one (decode full, max_bs 64)")
mc = ModelConfig.from_server_args(sa)
spec, reasons = g.megakernel_disable_reasons(cfg=cfg, hf_config=mc.hf_config, isa="sm_100")
reasons += g._dtype_reasons(spec, mc)
print("spec:", getattr(spec, "name", None))
print("REFUSALS:" if reasons else "NO REFUSAL: the command line passes every framework check")
for r in reasons:
    print("  -", r)
sys.exit(1 if reasons else 0)
