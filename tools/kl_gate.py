"""Numerics gate before opening traffic: NVIDIA's own KL test (test_covered_steps_match_prefill in sgl-nvidia-mega)
against a RUNNING server. Every token the megakernel's covered steps generate is scored again by the normal prefill
path; mean KL must be <= 0.03 (NVIDIA's bound). Ours on 2026-10-20: 0.0044 (8 prompts x 256 tokens).
Works with --api-key and --admin-api-key: calls get API_KEY, /flush_cache (an admin route) gets ADMIN_API_KEY.
Usage: URL=http://127.0.0.1:8000 API_KEY=... [ADMIN_API_KEY=...] MODEL_PATH=... LONGBENCH_CACHE=dir \
       PYTHONPATH=<overlay> <image python> tools/kl_gate.py
"""
import os
import sys

import requests

USER = os.environ.get("API_KEY", "")
ADMIN = os.environ.get("ADMIN_API_KEY", USER)
URL = os.environ.get("URL", "http://127.0.0.1:8000")
_post, _get = requests.post, requests.get


def _hdr(url, kw):
    h = dict(kw.pop("headers", None) or {})
    h["Authorization"] = "Bearer " + (ADMIN if "/flush_cache" in url else USER)
    kw["headers"] = h
    return kw


def _post_auth(url, *a, **kw):
    r = _post(url, *a, **_hdr(url, dict(kw)))
    if r.status_code == 401 and "/flush_cache" in url:  # server without --admin-api-key: the user key works
        h = dict(kw.pop("headers", None) or {}); h["Authorization"] = "Bearer " + USER
        r = _post(url, *a, headers=h, **kw)
    return r


requests.post = _post_auth
requests.get = lambda url, *a, **kw: _get(url, *a, **_hdr(url, kw))

import sglang.test.kl_test_utils as k  # noqa: E402

if os.environ.get("LONGBENCH_CACHE"):
    k.CACHE_DIR = os.environ["LONGBENCH_CACHE"]
MODEL = os.environ.get("MODEL_PATH", "/models/dsv41-1017")
try:
    k.test_input_output_logprobs_match_helper(URL, {MODEL: {"kl_div": 0.03}}, MODEL,
                                              max_samples=8, max_new_tokens=256, trust_remote_code=True)
except AssertionError as e:
    print("KL FAILED:", str(e)[:400])
    sys.exit(1)
print("KL OK (<= 0.03)")
