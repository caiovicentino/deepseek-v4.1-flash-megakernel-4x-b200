"""Quick, paired GSM8K for comparing two servers (stdlib only). The first N questions of the official test set
(openai/grade-school-math test.jsonl), chat without reasoning, temperature 0, max_tokens 1200, fixed concurrency.
Writes one JSONL row per question (id, gold, extracted answer, correct) so two runs can be paired (McNemar).
Not a capability eval: a guard against numerical breakage. Our megakernel production gate: 491/500 = 0.982.
Usage: python3 gsm8k_quick.py --url http://127.0.0.1:8000 --out g.jsonl [--key-file KEYFILE] [--n 500] [--conc 32]
       [--data test.jsonl]  (download: https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl)
"""
import argparse
import concurrent.futures as cf
import json
import re
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--key-file")
ap.add_argument("--model", default="glm-5.3-flash")
ap.add_argument("--data", default="gsm8k_test.jsonl")
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--conc", type=int, default=32)
a = ap.parse_args()
KEY = open(a.key_file).read().strip() if a.key_file else ""
H = {"Content-Type": "application/json", **({"Authorization": "Bearer " + KEY} if KEY else {})}
INSTR = ('Solve the following math problem step by step. The last line of your response should be of the form '
         '"#### ANSWER" where ANSWER is the final number.\n\n')
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def numero(s):
    if s is None:
        return None
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", s)
    cand = m[-1] if m else (NUM.findall(s) or [None])[-1]
    if cand is None:
        return None
    try:
        return float(cand.replace(",", "").rstrip("."))
    except ValueError:
        return None


def uma(item):
    i, q = item
    body = {"model": a.model, "messages": [{"role": "user", "content": INSTR + q["question"]}], "max_tokens": 1200,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    try:
        r = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=H)
        d = json.loads(urllib.request.urlopen(r, timeout=600).read())
        msg = d["choices"][0]["message"]
        txt = msg.get("content") or ""
        erro = None
    except Exception as e:  # noqa: BLE001
        txt, erro = "", f"{type(e).__name__}: {str(e)[:200]}"
    ouro = float(q["answer"].split("####")[-1].strip().replace(",", ""))
    resp = numero(txt)
    return {"id": i, "ouro": ouro, "resp": resp, "acerto": resp is not None and abs(resp - ouro) < 1e-6,
            "erro": erro, "n_chars": len(txt)}


qs = [json.loads(l) for l in open(a.data)][: a.n]
with cf.ThreadPoolExecutor(a.conc) as ex:
    res = sorted(ex.map(uma, enumerate(qs)), key=lambda x: x["id"])
with open(a.out, "w") as f:
    for r in res:
        f.write(json.dumps(r) + "\n")
ac = sum(r["acerto"] for r in res)
err = sum(1 for r in res if r["erro"])
print(f"GSM8K {a.url}: {ac}/{len(res)} = {ac / len(res):.3f} (call errors: {err})")
