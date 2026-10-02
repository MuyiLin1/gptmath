"""How asymmetric are trained query-key matrices?  (App. "Asymmetry of trained
query-key matrices", intro paragraph 3)

Four models: GPT-2 small and SmolLM2-360M (decoders), BERT-base-uncased and
ModernBERT-base (encoders).  For every layer and query head h the attention score is
score(x, x') = x^T M_h x' with the d x d bilinear form (d = model width: 768, or 960
for SmolLM2)
    M_h = W_Q,h W_K,h^T / sqrt(d_head)      (W_Q,h, W_K,h: d x d_head slices),
not the d_head x d_head product W_Q,h^T W_K,h.  We report
    r = ||(M - M^T)/2||_F / ||(M + M^T)/2||_F       (Frobenius norms; scale-free),
so r = 0 means perfectly symmetric and r ~ 1 is what random matrices of the same shape
give (printed as a baseline for each width).  Biases are ignored.
Grouped-query attention (SmolLM2): query head h uses key head h // (n_heads / n_kv_heads).
Rotary position embeddings (ModernBERT, SmolLM2) are ignored, i.e. this is the score of
two tokens at the same position, where the rotation cancels.
Secondary column: the same ratio after folding in the norm gain g that precedes
attention (LayerNorm or RMSNorm), M -> diag(g) M diag(g).

Needs `transformers` (optional dependency) and downloads ~2.3 GB of weights from the
Hugging Face hub on first run.  To use local copies instead, set QK_MODEL_DIR to a folder
containing gpt2/, bert-base-uncased/, ModernBERT-base/ and SmolLM2-360M/.
Run from inside export/:  PYTHONPATH=. python verification/qk_asymmetry_check.py
"""
import os
import numpy as np
import torch
from transformers import AutoModel

MODEL_DIR = os.environ.get("QK_MODEL_DIR")
HUB = {"gpt2": "gpt2", "bert-base-uncased": "bert-base-uncased",
       "ModernBERT-base": "answerdotai/ModernBERT-base", "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M"}


def load(name):
    return AutoModel.from_pretrained(os.path.join(MODEL_DIR, name) if MODEL_DIR else HUB[name]).eval()


def np_(t):
    return t.detach().double().numpy()


def ratio(M, d):
    assert M.shape == (d, d), (M.shape, d)
    S = 0.5 * (M + M.T)
    K = 0.5 * (M - M.T)
    return float(np.linalg.norm(K, "fro") / np.linalg.norm(S, "fro"))


def forms(Wq, Wk, g, n_heads, n_kv, dh, d, layer):
    """Wq: (n_heads*dh, d), Wk: (n_kv*dh, d) in torch Linear layout (q = Wq x)."""
    group = n_heads // n_kv
    out = []
    for h in range(n_heads):
        kv = h // group
        q, k = Wq[h * dh:(h + 1) * dh, :], Wk[kv * dh:(kv + 1) * dh, :]   # d_head x d each
        M = q.T @ k / np.sqrt(dh)                                          # d x d, score = x^T M x'
        out.append((layer, h, ratio(M, d), ratio(g[:, None] * M * g[None, :], d)))
    return out


def gpt2_forms():
    m = load("gpt2"); c = m.config
    d, H = c.n_embd, c.n_head; dh = d // H
    out = []
    for li, blk in enumerate(m.h):
        W = np_(blk.attn.c_attn.weight)                   # (d, 3d), Conv1D: q = x @ W[:, :d]
        out += forms(W[:, :d].T, W[:, d:2 * d].T, np_(blk.ln_1.weight), H, H, dh, d, li)
    return out, d


def bert_forms():
    m = load("bert-base-uncased"); c = m.config
    d, H = c.hidden_size, c.num_attention_heads; dh = d // H
    out = []
    for li, layer in enumerate(m.encoder.layer):
        # post-LN: the input to layer li is the LayerNorm output of the previous block
        g = np_(m.embeddings.LayerNorm.weight if li == 0 else m.encoder.layer[li - 1].output.LayerNorm.weight)
        out += forms(np_(layer.attention.self.query.weight), np_(layer.attention.self.key.weight),
                     g, H, H, dh, d, li)
    return out, d


def modernbert_forms():
    m = load("ModernBERT-base"); c = m.config
    d, H = c.hidden_size, c.num_attention_heads; dh = d // H
    out = []
    for li, layer in enumerate(m.layers):
        W = np_(layer.attn.Wqkv.weight)                   # (3d, d): rows = [Q | K | V], head-major
        # pre-norm; layer 0 has no attn_norm, its input is the embedding norm output
        g = np_(m.embeddings.norm.weight if li == 0 else layer.attn_norm.weight)
        out += forms(W[:d], W[d:2 * d], g, H, H, dh, d, li)
    return out, d


def smollm2_forms():
    m = load("SmolLM2-360M"); c = m.config
    d, H, KV = c.hidden_size, c.num_attention_heads, c.num_key_value_heads
    dh = getattr(c, "head_dim", None) or d // H
    out = []
    for li, layer in enumerate(m.layers):
        out += forms(np_(layer.self_attn.q_proj.weight), np_(layer.self_attn.k_proj.weight),
                     np_(layer.input_layernorm.weight), H, KV, dh, d, li)
    return out, d


def baseline(d, dh=64, n=200):
    rng = np.random.default_rng(0)
    return [ratio(rng.normal(size=(d, dh)) @ rng.normal(size=(d, dh)).T, d) for _ in range(n)]


def summarize(name, rows, d):
    r = np.array([x[2] for x in rows])
    rg = np.array([x[3] for x in rows])
    print(f"\n{name}: {len(rows)} heads, M is {d} x {d}, Frobenius norms")
    print(f"  r (as specified)        median {np.median(r):.3f}   range {r.min():.3f}-{r.max():.3f}"
          f"   IQR {np.percentile(r, 25):.3f}-{np.percentile(r, 75):.3f}")
    print(f"  r with norm gain        median {np.median(rg):.3f}   range {rg.min():.3f}-{rg.max():.3f}")
    layers = sorted({x[0] for x in rows})
    print("  per-layer median r: " + " ".join(f"{np.median([x[2] for x in rows if x[0] == l]):.2f}"
                                             for l in layers))
    return r


torch.set_grad_enabled(False)
for d in (768, 960):
    b = baseline(d)
    print(f"random baseline (200 draws of a Gaussian {d}x64 times 64x{d}): "
          f"r median {np.median(b):.3f}, range {min(b):.3f}-{max(b):.3f}")
res = {}
for name, fn in [("GPT-2 small (decoder)", gpt2_forms), ("BERT-base-uncased (encoder)", bert_forms),
                 ("ModernBERT-base (encoder)", modernbert_forms), ("SmolLM2-360M (decoder)", smollm2_forms)]:
    rows, d = fn()
    res[name.split()[0]] = summarize(name, rows, d)
rg, rb, rm, rs = res["GPT-2"], res["BERT-base-uncased"], res["ModernBERT-base"], res["SmolLM2-360M"]
print(f"\nGPT-2 median r / BERT median r = {np.median(rg) / np.median(rb):.2f}")
print(f"fraction of GPT-2 heads above BERT's median: {np.mean(rg > np.median(rb)):.2f}")
print(f"SmolLM2 median r / ModernBERT median r = {np.median(rs) / np.median(rm):.2f}")
print(f"fraction of SmolLM2 heads above ModernBERT's median: {np.mean(rs > np.median(rm)):.2f}")
