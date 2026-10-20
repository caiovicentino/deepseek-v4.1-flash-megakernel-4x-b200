# Patches

| File | What | Upstream |
|---|---|---|
| `sglang-pr44945-applied.diff` | SGLang's megakernel framework: `--enable-nvidia-megakernel`, plugin contract, guards, runtime, binding, DSpark draft-sampler hook, and its unit tests (15 files) | [sgl-project/sglang#44945](https://github.com/sgl-project/sglang/pull/44945), open. Drop this once it is merged |
| `sgl-nvidia-mega/0001-...` | Accept the B200: `isas={"sm_103","sm_100"}`, `resident_blocks=148` | ours; see [`../docs/porting-to-b200.md`](../docs/porting-to-b200.md) |
| `sgl-nvidia-mega/0002-...` | One unit-test case that asserted `sm_100` is refused now uses `sm_90` | ours |

`apply.sh <sglang tree> <sgl-nvidia-mega clone>` applies all three. It checks out `sgl-nvidia-mega` at `d47deb3` first.

You also need the base recipe's two patches (thinking rule; tool-result image order):
https://github.com/caiovicentino/deepseek-v4.1-flash-sglang-4x-b200/tree/main/patches
