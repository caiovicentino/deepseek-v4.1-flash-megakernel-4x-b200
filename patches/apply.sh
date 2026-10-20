#!/usr/bin/env bash
# Applies the two pieces of this recipe:
#  1) SGLang's megakernel framework: the python/ part of sgl-project/sglang PR #44945 (head 5ba9bb2dc2) and its unit
#     tests. It applied cleanly on SGLang ed8c3139 (image lmsysorg/sglang:nightly-dev-cu13-20261016-a93f16f0) with the
#     two patches of the base recipe already in. Once #44945 is merged upstream, skip this step.
#  2) sgl-nvidia-mega at d47deb3 + our B200 port (2 commits: accept sm_100 with 148 SMs; one unit test updated).
# Usage: bash patches/apply.sh <SGLang source tree, the dir that holds python/> <sgl-nvidia-mega git clone>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SGL=$1; MEGA=$2
(cd "$SGL" && git apply --check "$HERE/sglang-pr44945-applied.diff" && git apply "$HERE/sglang-pr44945-applied.diff")
echo "SGLang: megakernel framework applied ($SGL/python/sglang/srt/megakernels)"
git -C "$MEGA" checkout -q -B b200-sm100 d47deb3
git -C "$MEGA" -c user.name=port -c user.email=port@local am -q "$HERE"/sgl-nvidia-mega/*.patch
echo "sgl-nvidia-mega: $(git -C "$MEGA" log --oneline -1)"
echo "Install the plugin where only the megakernel server sees it, e.g.:"
echo "  pip install --no-deps --target /opt/mega-site $MEGA   and serve with SGLANG_OVERLAY=/opt/mega-site:$SGL/python"
