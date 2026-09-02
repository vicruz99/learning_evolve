#!/usr/bin/env python3
"""Minimal OpenAI-compatible embedding server on CPU, for ShinkaEvolve's duplicate gate.

Marvin compute nodes have no internet and the job's single GPU belongs to the kernel grader
(exclusive-process: a second CUDA context would lock the grader out), so the embedder runs on
the job's CPU cores from a local snapshot of Qwen/Qwen3-Embedding-0.6B. Shinka only embeds one
candidate program per generation, so ~1 s per call on 8 cores is fine.

    ~/venvs/embed/bin/python embed_server.py --model ~/models/qwen3-embedding-0.6b --port 8003

Shinka side:  --embedding_model "local/Qwen/Qwen3-Embedding-0.6B@http://127.0.0.1:8003/v1"

Endpoints: POST /v1/embeddings (OpenAI shape), GET /v1/models, GET /health.
Pooling follows the model card: last-token hidden state, L2-normalised.
"""
import argparse
import os
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI()
STATE = {}


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str | None = None
    dimensions: int | None = None


@torch.inference_mode()
def embed(texts: list[str]) -> list[list[float]]:
    tok, model, max_len = STATE["tok"], STATE["model"], STATE["max_len"]
    batch = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    out = model(**batch).last_hidden_state
    # last non-pad token (left or right padding both handled)
    if tok.padding_side == "left":
        vec = out[:, -1]
    else:
        lengths = batch["attention_mask"].sum(dim=1) - 1
        vec = out[torch.arange(out.shape[0]), lengths]
    vec = torch.nn.functional.normalize(vec.float(), p=2, dim=1)
    return vec.tolist()


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    t0 = time.time()
    vecs = embed(texts)
    n_tok = sum(len(STATE["tok"](t)["input_ids"]) for t in texts)
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
        "model": STATE["name"],
        "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok},
        "_elapsed_s": round(time.time() - t0, 3),
    }


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": STATE["name"], "object": "model", "owned_by": "local"}]}


@app.get("/health")
def health():
    return {"status": "ok", "model": STATE["name"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/qwen3-embedding-0.6b"),
                   help="local snapshot directory (no hub access is attempted)")
    p.add_argument("--served-name", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--threads", type=int, default=int(os.environ.get("EMBED_THREADS", "4")))
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModel.from_pretrained(a.model, local_files_only=True, torch_dtype=torch.float32).eval()
    STATE.update(tok=tok, model=model, max_len=a.max_len, name=a.served_name)
    print(f"[embed] {a.served_name} from {a.model} on CPU ({a.threads} threads), :{a.port}", flush=True)

    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
