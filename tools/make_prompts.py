"""Builds the benchmark prompts as TOKEN-ID LISTS (CPU only), so two engines receive exactly the same tokens in
/v1/completions and no chat-template difference enters the measurement. Rendered with the checkpoint's own reference
encoder (encoding/encoding.py of DeepSeek-V4.1-Flash-1017), chat mode. Context = real Python source files in a fixed
order, like an agentic coding session; within a context all requests share the prefix (prefix-cache hits, like our
traffic) and the question changes. Contexts: 8k, 64k and "lm" (~2.85k tokens + a coding task: the LocalMaxxing-style
single-stream test). The questions are in Portuguese, as our users write.
Usage: python -I make_prompts.py prompts.json [MODEL_PATH] [SOURCE_TREE]
"""
import hashlib
import importlib.util
import json
import pathlib
import sys

MODEL = sys.argv[2] if len(sys.argv) > 2 else "/models/dsv41-1017"
SRC = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else "vllm/vllm")  # any Python source tree; we used vLLM's
ALVOS = {"8k": 8_000, "64k": 64_000, "lm": 3_000}
PERGUNTAS = [
    "Explique em detalhes, em português, como os módulos acima se relacionam, função por função.",
    "Faça uma revisão de código minuciosa do trecho acima, em português, apontando riscos e melhorias.",
    "Escreva, em português, um guia de estudo longo e completo para um engenheiro novo entender esse código.",
    "Descreva, em português e com exemplos, o fluxo de dados de ponta a ponta no código acima.",
    "Liste e explique, em português, todas as decisões de projeto que você consegue inferir desse código.",
    "Escreva testes unitários em Python para as funções principais acima, explicando cada caso em português.",
    "Proponha, em português, uma refatoração completa do código acima, com o código novo e a justificativa.",
    "Explique, em português, os pontos de concorrência e de desempenho do código acima, com detalhes.",
]

spec = importlib.util.spec_from_file_location("enc_ref", f"{MODEL}/encoding/encoding.py")
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)
from transformers import PreTrainedTokenizerFast  # noqa: E402

tok = PreTrainedTokenizerFast(tokenizer_file=f"{MODEL}/tokenizer.json")
arquivos = sorted(p for p in SRC.rglob("*.py") if "third_party" not in p.parts and p.stat().st_size > 2000)


def contexto(alvo: int) -> str:
    partes, n = [], 0
    for p in arquivos:
        t = f"### Arquivo: {p.relative_to(SRC.parent)}\n```python\n{p.read_text(errors='ignore')}\n```\n\n"
        k = len(tok.encode(t, add_special_tokens=False))
        if n + k > alvo:
            continue  # arquivo grande demais para o que falta: tenta o próximo
        partes.append(t)
        n += k
        if n > 0.98 * alvo:
            break
    return "".join(partes)


PERGUNTAS_LM = [  # protocolo do LocalMaxxing: ~2,7k de entrada, saída de código, lote 1
    "Implemente em Python, do zero, uma árvore rubro-negra completa com inserção, remoção e busca, com testes unitários.",
    "Escreva em Python um servidor HTTP assíncrono com roteamento, middlewares e testes, sem bibliotecas externas.",
    "Implemente em Python um interpretador para uma linguagem de expressões aritméticas com variáveis e funções, com testes.",
    "Escreva em Python um cache LRU thread-safe com expiração por tempo, métricas e testes unitários completos.",
    "Implemente em Python um parser de JSON do zero, com mensagens de erro com linha e coluna, e testes.",
    "Escreva em Python um agendador de tarefas com prioridades, dependências e detecção de ciclos, com testes.",
    "Implemente em Python um índice invertido com busca booleana e ranking BM25, com testes unitários.",
    "Escreva em Python um banco chave-valor com log de escrita antecipada e recuperação após falha, com testes.",
]
saida = {"modelo": MODEL, "encoder": "encoding/encoding.py do checkpoint (modo chat)", "contextos": {}}
for nome, alvo in ALVOS.items():
    ctx = contexto(alvo)
    lista = []
    for q in (PERGUNTAS_LM if nome == "lm" else PERGUNTAS):
        msgs = [{"role": "user", "content": ctx + "\n" + q}]
        texto = enc.encode_messages(msgs, thinking_mode="chat")
        ids = tok.encode(texto, add_special_tokens=False)
        lista.append(ids)
    comum = 0
    while all(len(x) > comum and x[comum] == lista[0][comum] for x in lista):
        comum += 1
    saida["contextos"][nome] = {"prompts": lista, "tokens": [len(x) for x in lista], "prefixo_comum": comum}
    print(nome, "tokens por prompt:", [len(x) for x in lista], "| prefixo comum:", comum)
    print("  início:", repr(tok.decode(lista[0][:12])), "| fim:", repr(tok.decode(lista[0][-12:])))
blob = json.dumps(saida, separators=(",", ":")).encode()
pathlib.Path(sys.argv[1]).write_bytes(blob)
print("gravado", sys.argv[1], len(blob), "bytes, sha256", hashlib.sha256(blob).hexdigest())
