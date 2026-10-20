"""Per-user decode benchmark, identical for any OpenAI-compatible server (we used it on SGLang and vLLM). Stdlib only.
Each cell = (context, concurrency c): c streams at once, each sends R requests in sequence; every request sends the
token-id list from prompts.json to /v1/completions (no chat template in the path, so two engines get identical
tokens), fixed max_tokens, ignore_eos (guard against early stops), temperature 0.6 / top_p 0.95, streaming with usage.
Per request: TTFT and per-user decode = (tokens - 1) / (last chunk - first chunk), tokens from `usage` (with
speculative decoding a chunk carries several tokens). /metrics speculative lines are saved before/after each cell.
Usage: python3 bench_engines.py --url http://127.0.0.1:8000 --label name --out res.jsonl [--key-file KEYFILE]
       [--cells "64k:1,2,4,8,16;8k:1,4,16,32,64"] [--passes 2] [--max-tokens 1024] [--prompts prompts.json]
LocalMaxxing-style single stream: --cells "lm:1" --max-tokens 2048 (the "lm" context is a ~2.85k-token coding prompt).
"""
import argparse
import json
import re
import statistics
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--key-file")
ap.add_argument("--model", default="glm-5.3-flash")
ap.add_argument("--prompts", default="prompts.json")
ap.add_argument("--cells", default="64k:1,2,4,8,16;8k:1,4,16,32,64")
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--timeout", type=int, default=600)
a = ap.parse_args()
KEY = open(a.key_file).read().strip() if a.key_file else ""
H = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}
P = json.load(open(a.prompts))["contextos"]
out = open(a.out, "a")
lock = threading.Lock()


def metricas():
    try:
        r = urllib.request.Request(a.url + "/metrics", headers=H)
        txt = urllib.request.urlopen(r, timeout=20).read().decode()
    except Exception as e:  # noqa: BLE001
        return {"erro": str(e)[:200]}
    linhas = [l for l in txt.splitlines() if l and not l.startswith("#") and re.search(r"spec|accept|draft|num_generation_tokens|generation_tokens_total", l)]
    return {"linhas": linhas}


def pedido(ids):
    body = {"model": a.model, "prompt": ids, "max_tokens": a.max_tokens, "temperature": 0.6, "top_p": 0.95,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    t1 = tN = None
    n_usage = None
    erro = None
    try:
        r = urllib.request.Request(a.url + "/v1/completions", data=json.dumps(body).encode(), headers=H)
        with urllib.request.urlopen(r, timeout=a.timeout) as resp:
            for raw in resp:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                j = json.loads(d)
                if j.get("usage"):
                    n_usage = j["usage"].get("completion_tokens")
                for ch in j.get("choices") or []:  # the usage chunk comes with choices: []
                    if ch.get("text"):
                        now = time.time()
                        t1 = t1 or now
                        tN = now
    except Exception as e:  # noqa: BLE001
        erro = f"{type(e).__name__}: {str(e)[:200]}"
    res = {"ttft_s": (t1 - t0) if t1 else None, "total_s": time.time() - t0, "tokens": n_usage, "erro": erro}
    if t1 and tN and n_usage and tN > t1:
        res["decode_tps"] = (n_usage - 1) / (tN - t1)
    return res


def celula(ctx, c, passe):
    reps = 3 if c <= 8 else 2
    prompts = P[ctx]["prompts"]
    m0 = metricas()
    t0 = time.time()
    regs = []

    def fluxo(i):
        for r in range(reps):
            ids = prompts[(i * reps + r) % len(prompts)]
            res = pedido(ids)
            res.update({"label": a.label, "ctx": ctx, "c": c, "passe": passe, "fluxo": i, "rep": r, "prompt_tokens": len(ids)})
            with lock:
                regs.append(res)
                out.write(json.dumps(res) + "\n")
                out.flush()

    th = [threading.Thread(target=fluxo, args=(i,)) for i in range(c)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    m1 = metricas()
    with lock:
        out.write(json.dumps({"label": a.label, "ctx": ctx, "c": c, "passe": passe, "tipo": "metricas", "antes": m0, "depois": m1, "dur_s": time.time() - t0}) + "\n")
        out.flush()
    ok = [x for x in regs if x.get("decode_tps") and not x["erro"]]
    errs = sum(1 for x in regs if x["erro"])
    if not ok:
        print(f"{a.label} {ctx} c={c} pass {passe}: NO valid result ({errs} errors)", flush=True)
        return
    dec = statistics.median(x["decode_tps"] for x in ok)
    tt = sorted(x["ttft_s"] for x in ok)
    print(f"{a.label} {ctx:>3} c={c:<3} pass {passe}: per-user decode median {dec:7.1f} tok/s | TTFT p50 {tt[len(tt)//2]:.2f} s "
          f"| tokens/request {statistics.median(x['tokens'] for x in ok):.0f} | n={len(ok)} errors={errs}", flush=True)


cells = [(ctx, int(c)) for blk in a.cells.split(";") for ctx, cs in [blk.split(":")] for c in cs.split(",")]
print(f"[{time.strftime('%H:%M:%S')}] {a.label}: warm-up (prefix cache + graphs)", flush=True)
for ctx in sorted({x for x, _ in cells}):
    pedido(P[ctx]["prompts"][0])
for passe in range(1, a.passes + 1):
    for ctx, c in cells:
        celula(ctx, c, passe)
print(f"[{time.strftime('%H:%M:%S')}] {a.label}: done", flush=True)
