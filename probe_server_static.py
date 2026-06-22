"""TrustSearch probe server -- STATIC (frozen probe heads, live encoder).

This is the ablation variant for the ts_static experiment:
- Probe heads (commit_probe, boundary_probe) are frozen from round-0 (base250b)
- Encoder (for extracting hidden states) tracks the current policy checkpoint
  (reloaded from the training manifest when new checkpoints appear)

This isolates the contribution of co-evolution: same probe signal quality as
"frozen r0 probe applied to current policy's representations" without refit.

Usage:
    python probe_server_static.py --manifest probes/live_ts_static/manifest.json \
        --frozen_commit probes/base250b/commit_probe.pkl \
        --frozen_boundary probes/base250b/boundary_probe.pkl \
        --port 8002
"""
from __future__ import annotations
import os
for _k in list(os.environ.keys()):
    if "proxy" in _k.lower():
        del os.environ[_k]

import argparse, json, re, threading, time
from typing import List
import numpy as np
import torch
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = torch.bfloat16
app = FastAPI()
LOCK = threading.Lock()
STATE = {"round": -1, "encoder": None, "model": None, "tok": None,
         "commit": None, "boundary": None}
MANIFEST = None
FROZEN_COMMIT = None
FROZEN_BOUNDARY = None


def _load_model(encoder):
    tok = AutoTokenizer.from_pretrained(encoder, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        encoder, torch_dtype=DTYPE, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    return tok, model


def _apply(manifest_path, force=False):
    """Watch manifest for new encoder checkpoints, but NEVER reload probes."""
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path) as f:
        m = json.load(f)
    if (not force) and m.get("round", -1) == STATE["round"]:
        return
    enc = m["encoder"]
    reload_model = force or (enc != STATE["encoder"])
    print(f"[probe-static] round={m.get('round')} encoder={enc} "
          f"(model_reload={reload_model}, probes=FROZEN)", flush=True)
    tok = STATE["tok"]; model = STATE["model"]
    if reload_model:
        tok, model = _load_model(enc)
    old = STATE["model"] if reload_model else None
    with LOCK:
        STATE.update({"round": m.get("round", STATE["round"] + 1), "encoder": enc,
                      "tok": tok, "model": model,
                      "commit": FROZEN_COMMIT, "boundary": FROZEN_BOUNDARY})
    if old is not None and old is not model:
        del old
        torch.cuda.empty_cache()
    print(f"[probe-static] serving round={STATE['round']} "
          f"commit=FROZEN(L{FROZEN_COMMIT['layer']}) "
          f"boundary=FROZEN(L{FROZEN_BOUNDARY['layer']})", flush=True)


def _watcher():
    while True:
        try:
            _apply(MANIFEST)
        except Exception as e:
            print(f"[probe-static] watcher error: {e}", flush=True)
        time.sleep(15)


class Item(BaseModel):
    prompt: str
    response: str = ""


class ProbeRequest(BaseModel):
    items: List[Item]


@torch.no_grad()
def _last_hidden(texts, layer, model, tok, micro=16, max_len=4096):
    base = model.model
    cap = {}
    h = base.layers[layer - 1].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    out = []
    try:
        for s in range(0, len(texts), micro):
            enc = tok(texts[s:s + micro], return_tensors="pt", add_special_tokens=False,
                      padding=True, truncation=True, max_length=max_len).to(base.device)
            base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 use_cache=False)
            out.append(cap["h"][:, -1, :].float().cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out) if out else np.zeros((0, 1))


def _commit_ctx(prompt, response):
    ae = response.rfind("</answer>")
    return prompt + (response[:ae] if ae >= 0 else response)


@app.post("/judge_probe")
def judge_probe(req: ProbeRequest):
    if not req.items:
        return {"results": []}
    with LOCK:
        model, tok = STATE["model"], STATE["tok"]
        commit, boundary = STATE["commit"], STATE["boundary"]
    cc = [_commit_ctx(it.prompt, it.response) for it in req.items]
    bc = [it.prompt for it in req.items]
    Hc = _last_hidden(cc, commit["layer"], model, tok)
    Hb = _last_hidden(bc, boundary["layer"], model, tok)
    flip = commit["clf"].predict_proba(Hc)[:, 1]
    bnd = boundary["clf"].predict_proba(Hb)[:, 1]
    return {"results": [{"flip": float(flip[i]), "boundary": float(bnd[i])}
                        for i in range(len(req.items))]}


@app.get("/health")
def health():
    return {"ok": STATE["model"] is not None, "round": STATE["round"],
            "encoder": STATE["encoder"], "mode": "STATIC (frozen probes)"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="Manifest JSON that tracks encoder checkpoint (probe fields ignored)")
    ap.add_argument("--frozen_commit", required=True,
                    help="Path to frozen r0 commit probe pkl")
    ap.add_argument("--frozen_boundary", required=True,
                    help="Path to frozen r0 boundary probe pkl")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()

    MANIFEST = args.manifest
    FROZEN_COMMIT = joblib.load(args.frozen_commit)
    FROZEN_BOUNDARY = joblib.load(args.frozen_boundary)
    print(f"[probe-static] FROZEN probes: commit(L{FROZEN_COMMIT['layer']}) "
          f"boundary(L{FROZEN_BOUNDARY['layer']})", flush=True)
    print(f"[probe-static] manifest={MANIFEST} waiting...", flush=True)

    for _ in range(600):
        if os.path.exists(MANIFEST):
            break
        time.sleep(2)
    _apply(MANIFEST, force=True)
    threading.Thread(target=_watcher, daemon=True).start()
    print(f"[probe-static] serving on 0.0.0.0:{args.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
