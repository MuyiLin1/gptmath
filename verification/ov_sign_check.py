"""E2: do trained attention heads pull tokens together (like V = +Id) or push them apart (like V = -Id)?
(Appendix "Asymmetry of trained query-key matrices", paragraph "Sign of the value-output map")

For every head h the update a token receives is W_O,h W_V,h applied to the (attention-weighted)
average it attends to.  Its nonzero eigenvalues are those of the d_h x d_h matrix
    P_h = W_V,h W_O,h        (torch Linear layout: W_V,h is d_h x d, W_O,h is d x d_h).
Positive real parts pull a token toward what it attends to; negative ones push it away.
Per head we report
    ratio = tr(P_h) / sum|lambda|      in [-1, 1]   (+1: pure pull, -1: pure push)
    push  = sum_{Re lambda < 0} |lambda| / sum|lambda|
and a second version with the gain g of the norm that precedes attention folded in,
P_h -> W_V,h diag(g) W_O,h.  Grouped-query attention (SmolLM2): query head h uses value head
h // (n_heads / n_kv).  Biases are ignored.  Baseline: Gaussian W_V, W_O of the same shape.

Weights are read straight from the safetensors files on the Hugging Face hub (~2.3 GB on first
run), so any recent torch + huggingface_hub + safetensors works; transformers is not needed.
Run from inside export/:  PYTHONPATH=. python verification/ov_sign_check.py
"""
import os
import re
import json
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "review"))
MODELS = {
    "GPT-2 small (decoder)": dict(repo="gpt2", L=12, H=12, KV=12, d=768),
    "BERT-base-uncased (encoder)": dict(repo="bert-base-uncased", L=12, H=12, KV=12, d=768),
    "ModernBERT-base (encoder)": dict(repo="answerdotai/ModernBERT-base", L=22, H=12, KV=12, d=768),
    "SmolLM2-360M (decoder)": dict(repo="HuggingFaceTB/SmolLM2-360M", L=32, H=15, KV=5, d=960),
}


def getter(sd):
    keys = list(sd.keys())

    def get(*suffixes, required=True):
        for suf in suffixes:
            hits = [k for k in keys if re.search(r"(^|\.)" + re.escape(suf) + r"$", k)]
            if len(hits) == 1:
                return sd[hits[0]].float().numpy().astype(np.float64)
        if required:
            raise KeyError(f"none of {suffixes} found; sample keys: {keys[:8]}")
        return None
    return get


def layer_mats(name, cfg, get, li):
    """Return Wv (KV*dh, d), Wo (d, H*dh), gain g (d,) in torch Linear layout."""
    d = cfg["d"]
    if name.startswith("GPT-2"):
        W = get(f"h.{li}.attn.c_attn.weight")                         # Conv1D (d, 3d): v = x @ W[:, 2d:]
        return W[:, 2 * d:].T, get(f"h.{li}.attn.c_proj.weight").T, get(f"h.{li}.ln_1.weight")
    if name.startswith("BERT"):
        g = (get("embeddings.LayerNorm.weight", "embeddings.LayerNorm.gamma") if li == 0 else
             get(f"layer.{li - 1}.output.LayerNorm.weight", f"layer.{li - 1}.output.LayerNorm.gamma"))
        return (get(f"layer.{li}.attention.self.value.weight"), get(f"layer.{li}.attention.output.dense.weight"), g)
    if name.startswith("ModernBERT"):
        W = get(f"layers.{li}.attn.Wqkv.weight")                       # (3d, d): rows [Q | K | V]
        g = get(f"layers.{li}.attn_norm.weight", required=False)
        if g is None:
            g = get("embeddings.norm.weight")
        return W[2 * d:], get(f"layers.{li}.attn.Wo.weight"), g
    return (get(f"layers.{li}.self_attn.v_proj.weight"), get(f"layers.{li}.self_attn.o_proj.weight"),
            get(f"layers.{li}.input_layernorm.weight"))


def head_stats(P):
    lam = np.linalg.eigvals(P)
    s = np.abs(lam).sum()
    return float(lam.real.sum() / s), float(np.abs(lam[lam.real < 0]).sum() / s)


def analyse(name, cfg):
    path = hf_hub_download(cfg["repo"], "model.safetensors")
    get = getter(load_file(path))
    H, KV, d = cfg["H"], cfg["KV"], cfg["d"]
    dh, group = d // H, H // KV
    rows = []
    for li in range(cfg["L"]):
        Wv, Wo, g = layer_mats(name, cfg, get, li)
        for h in range(H):
            kv = h // group
            Wv_h, Wo_h = Wv[kv * dh:(kv + 1) * dh], Wo[:, h * dh:(h + 1) * dh]
            r, p = head_stats(Wv_h @ Wo_h)
            rg, pg = head_stats(Wv_h @ (g[:, None] * Wo_h))
            rows.append(dict(layer=li, head=h, ratio=r, push=p, ratio_gain=rg, push_gain=pg))
    return rows


def summarize(name, rows):
    r = np.array([x["ratio"] for x in rows]); p = np.array([x["push"] for x in rows])
    rg = np.array([x["ratio_gain"] for x in rows])
    s = dict(heads=len(rows), frac_negative_trace=float(np.mean(r < 0)), frac_push_dominant=float(np.mean(p > 0.5)),
             frac_strong_push=float(np.mean(r < -0.2)), frac_strong_pull=float(np.mean(r > 0.2)),
             ratio_median=float(np.median(r)), ratio_q25=float(np.percentile(r, 25)), ratio_q75=float(np.percentile(r, 75)),
             ratio_min=float(r.min()), ratio_max=float(r.max()),
             frac_negative_trace_gain=float(np.mean(rg < 0)), ratio_median_gain=float(np.median(rg)))
    print(f"\n{name}: {s['heads']} heads", flush=True)
    print(f"  tr/sum|lam|: median {s['ratio_median']:+.3f}  IQR {s['ratio_q25']:+.3f}..{s['ratio_q75']:+.3f}  "
          f"range {s['ratio_min']:+.3f}..{s['ratio_max']:+.3f}")
    print(f"  heads with negative trace (push): {s['frac_negative_trace']:.1%}   push-dominant (push>0.5): "
          f"{s['frac_push_dominant']:.1%}   strong push (ratio<-0.2): {s['frac_strong_push']:.1%}   "
          f"strong pull (ratio>0.2): {s['frac_strong_pull']:.1%}")
    print(f"  with norm gain: negative trace {s['frac_negative_trace_gain']:.1%}, median ratio {s['ratio_median_gain']:+.3f}")
    layers = sorted({x["layer"] for x in rows})
    print("  per-layer fraction of push heads: " + " ".join(
        f"{np.mean([x['ratio'] < 0 for x in rows if x['layer'] == l]):.2f}" for l in layers), flush=True)
    return s


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_grad_enabled(False)
    rng = np.random.default_rng(0)
    base = [head_stats(rng.normal(size=(64, 768)) @ rng.normal(size=(768, 64)))[0] for _ in range(500)]
    print(f"random baseline (Gaussian W_V, W_O, d=768, d_h=64): ratio median {np.median(base):+.3f}, "
          f"IQR {np.percentile(base, 25):+.3f}..{np.percentile(base, 75):+.3f}, negative trace {np.mean(np.array(base) < 0):.1%}",
          flush=True)
    out = {"baseline_ratio": base}
    for i, (name, cfg) in enumerate(MODELS.items(), 1):
        print(f"[E2] {i}/{len(MODELS)} loading {cfg['repo']} ...", flush=True)
        try:
            rows = analyse(name, cfg)
            out[name] = dict(summary=summarize(name, rows), heads=rows)
        except Exception as e:                     # keep going so one model cannot block the others
            print(f"  FAILED for {name}: {type(e).__name__}: {e}", flush=True)
            out[name] = dict(error=f"{type(e).__name__}: {e}")
    with open(os.path.join(OUT, "E2_ov_sign.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved results/review/E2_ov_sign.json", flush=True)


if __name__ == "__main__":
    main()
