"""Does every CUDA kernel of sgl-nvidia-mega compile for the B200 (sm_100a)? CPU only: NVRTC needs no GPU.
Compiles each .cu with the SAME options and -D defines the plugin uses, for sm_103a (control: the original target)
and sm_100a (B200). A failure only on sm_100a would mean an SM103-only instruction. Result on 2026-10-19 (NVRTC 13.0):
34/34 valid variants compile for sm_100a; the 4 "fail on both" are combinations the code forbids with a static_assert.
Usage: python -I check_nvrtc_sm100.py <sgl-nvidia-mega clone>   (CUBIN_DIR=dir also saves the cubins, for
cuobjdump -res-usage)
"""
import itertools
import os
import pathlib
import sys

from cuda.bindings import nvrtc

ROOT = pathlib.Path(sys.argv[1]) / "sgl_nvidia_mega/deepseek_v41/sm103"
INCLUDES = [p for p in ["/usr/local/cuda/include", *sorted(str(p) for p in pathlib.Path(sys.prefix).glob("lib/python3*/site-packages/nvidia/*/include"))] if os.path.isdir(p)]


def compila(src: pathlib.Path, arch: str, options: tuple, headers: tuple = ()) -> tuple[bool, str, int]:
    hdr = {h: (src.parent / h).read_text() for h in headers}
    err, prog = nvrtc.nvrtcCreateProgram(src.read_bytes(), src.name.encode(), len(hdr),
                                         [v.encode() for v in hdr.values()], [k.encode() for k in hdr])
    opts = [f"--gpu-architecture={arch}", *options, *[f"-I{d}" for d in INCLUDES]]
    (st,) = nvrtc.nvrtcCompileProgram(prog, len(opts), [o.encode() for o in opts])
    _, n = nvrtc.nvrtcGetProgramLogSize(prog)
    log = b" " * n
    nvrtc.nvrtcGetProgramLog(prog, log)
    size = 0
    if int(st) == 0:
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cub = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cub)
        if os.environ.get("CUBIN_DIR"):
            tag = "_".join(o[2:].replace("=", "") for o in options if o.startswith("-D"))
            out = pathlib.Path(os.environ["CUBIN_DIR"]) / f"{src.stem}__{tag or 'base'}__{arch}.cubin"
            out.write_bytes(cub)
    nvrtc.nvrtcDestroyProgram(prog)
    return int(st) == 0, log.decode(errors="replace").strip(), size


MOE = ("-std=c++17", "-default-device", "--use_fast_math")
GATE = ("-std=c++17", "-default-device", "--ftz=true", "--prec-sqrt=false", "--prec-div=false")
casos = [
    ("moe/route", ROOT / "moe/src/route.cu", MOE, ("route.cuh",)),
    ("moe/route_wide", ROOT / "moe/src/route_wide.cu", MOE, ()),
    ("moe/gemm1", ROOT / "moe/src/gemm1.cu", MOE, ()),
    ("moe/gemm2", ROOT / "moe/src/gemm2.cu", MOE, ()),
    ("moe/gate_topk", ROOT / "moe/src/gate_topk.cu", GATE, ("gate_topk.cuh",)),
    ("moe/gate_topk_draft", ROOT / "moe/src/gate_topk_draft.cu", GATE, ("gate_topk.cuh",)),
]
for k, sh, nm, q in itertools.product((6, 8), (0, 1), (0, 1), (0, 1)):
    casos.append((f"two_shot_mhc k{k} sh{sh} n{nm} q{q}", ROOT / "src/two_shot_mhc.cu",
                  ("-std=c++17", "-DTS_WORLD=4", f"-DTS_TOP_K={k}", f"-DTS_HAS_SHARED={sh}", f"-DTS_NORM={nm}", f"-DTS_QUANT={q}"), ()))
for direct, steps in itertools.product((0, 1), (5, 7)):
    casos.append((f"markov direct{direct} steps{steps}", ROOT / "markov.cu",
                  ("-std=c++17", "-default-device", "-DBASE_T=__nv_bfloat16", "-DCACHE_T=float", f"-DMARKOV_DIRECT={direct}", f"-DSTEPS={steps}"), ()))
for tile, smem in ((16, 87040), (32, 107520), (48, 115712)):
    for i64, et in itertools.product((0, 1), (0, 1)):
        casos.append((f"wo_a m{tile} i64{i64} et{et}", ROOT / "attention/src/wo_a_fused.cu",
                      ("-std=c++20", "-default-device", f"-DWO_A_MAX_TOKENS={tile}", f"-DWO_A_POS_INT64={i64}",
                       f"-DWO_A_EARLY_TRIGGER={et}", "-DWO_A_ROW_TILE_SCALES=0", f"-DWO_A_SMEM_BYTES={smem}"), ()))

print("NVRTC", nvrtc.nvrtcVersion()[1:], "| includes:", len(INCLUDES))
so_103 = []
for nome, src, opts, hdrs in casos:
    r = {}
    for arch in ("sm_103a", "sm_100a"):
        ok, log, size = compila(src, arch, opts, hdrs)
        r[arch] = (ok, log, size)
    a, b = r["sm_103a"], r["sm_100a"]
    status = "OK on both" if a[0] and b[0] else ("ONLY sm_103a (SM103-only!)" if a[0] else ("ONLY sm_100a?" if b[0] else "FAILS on both (harness)"))
    print(f"{nome:34s} {status:32s} cubin 103={a[2]:>8} 100={b[2]:>8}")
    if a[0] and not b[0]:
        so_103.append(nome)
        print("    sm_100a log:", b[1][:600].replace("\n", " | "))
    if not a[0] and not b[0]:
        print("    sm_103a log:", a[1][:400].replace("\n", " | "))
print("SUMMARY:", f"{len(casos) - len(so_103)}/{len(casos)} variants compile for sm_100a;", "SM103-only:", so_103 or "none")
