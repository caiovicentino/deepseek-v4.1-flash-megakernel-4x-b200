# Third-party notices

- **sgl-nvidia-mega**, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, Apache License 2.0
  (https://github.com/sgl-project/sgl-nvidia-mega). The patches in `patches/sgl-nvidia-mega/` modify it, are
  distributed under the same license, and keep its copyright headers.
- **SGLang**, Apache License 2.0 (https://github.com/sgl-project/sglang). `patches/sglang-pr44945-applied.diff`
  is the python/ part and the unit tests of pull request #44945 (head 5ba9bb2dc2) by its authors, as applied to
  SGLang `ed8c3139`; it is redistributed unchanged under Apache-2.0.
- `tools/kl_gate.py` calls SGLang's `sglang.test.kl_test_utils`, and `tools/gates.sh` follows the e2e test in
  sgl-nvidia-mega; no code from them is copied.

Everything else in this repository is MIT (see LICENSE).
