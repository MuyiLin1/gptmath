"""E11: does the antisymmetric part change where tokens go over a realistic depth?
(Appendix "Trained heads as dynamical systems")

For every head of GPT-2 small and BERT-base, with the matrices and real token states of trained_head_flow.py
(cache results/review/E7_heads.npz), we iterate the layer map of one head,
    u_i <- normalize(u_i + h_i v_i(u)),   v = P_u( sum_j a_ij(D) V u_j ),   V = the head's value map / ||V||_2,
for L = 1..48 layers.  The step h_i is the head's real relative update of token i: one layer of the trained
head adds sqrt(d) ||V||_2 sum_j a_ij V u_j to the residual vector x_i, so h_i = sqrt(d) ||V||_2 / |x_i - c_i|
(c_i = mean of x_i for GPT-2, the layer-norm bias for BERT), from the real hidden states.  Biases and
centering of the update are ignored, so h is approximate.
Three attention matrices: the real D, its symmetric part D_s, and D_s plus a random antisymmetric part of
the same Frobenius norm (as in antisym_share.py).  Recorded after L in LREC layers: for real vs D_s and for
random vs D_s, the mean angle between the two copies of a token, the same relative to how far the token
moved, and the fraction of tokens whose nearest neighbour differs; and the mean pairwise cosine.
Run from inside export/ in the env of trained_head_flow.py (the step sizes need the model):
    PYTHONPATH=. python verification/antisym_depth.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "review"))
CACHE = os.path.join(OUT, "E7_heads.npz")
STEPS = os.path.join(OUT, "E11_steps.npz")
LREC = [1, 2, 3, 6, 12, 24, 48]
N = 64


def build_steps():
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from transformers import AutoModel, AutoTokenizer
    from trained_head_flow import PASSAGES
    torch.set_grad_enabled(False)
    d, H, dh = 768, 12, 64
    out = {}
    for model in ["gpt2", "bert-base-uncased"]:
        sd = {k.replace("bert.", "").replace("transformer.", ""): v.astype(np.float64)
              for k, v in load_file(hf_hub_download(model, "model.safetensors")).items()}
        tok, net = AutoTokenizer.from_pretrained(model), AutoModel.from_pretrained(model).eval()
        hs = []
        for p in PASSAGES:
            ids = tok(p, return_tensors="pt")["input_ids"]
            lo = 0 if model == "gpt2" else 1
            hs.append([h[0, lo:lo + N].double().numpy() for h in net(ids, output_hidden_states=True).hidden_states])
        for l in range(12):
            if model == "gpt2":
                W = sd[f"h.{l}.attn.c_attn.weight"]
                Wv, Wo, g, b = W[:, 2 * d:], sd[f"h.{l}.attn.c_proj.weight"], sd[f"h.{l}.ln_1.weight"], None
            else:
                pre = "embeddings.LayerNorm" if l == 0 else f"encoder.layer.{l - 1}.output.LayerNorm"
                g = sd.get(f"{pre}.weight", sd.get(f"{pre}.gamma"))
                b = sd.get(f"{pre}.bias", sd.get(f"{pre}.beta"))
                Wv = sd[f"encoder.layer.{l}.attention.self.value.weight"].T
                Wo = sd[f"encoder.layer.{l}.attention.output.dense.weight"].T
            radius = [np.linalg.norm(P[l] - (P[l].mean(1, keepdims=True) if b is None else b), axis=1) for P in hs]
            for h in range(H):
                sl = slice(h * dh, (h + 1) * dh)
                s = np.linalg.norm(Wo[sl].T @ (g[:, None] * Wv[:, sl]).T, 2)
                out[f"{model}|{l}|{h}"] = np.stack([np.sqrt(d) * s / r for r in radius])
    np.savez(STEPS, **out)


def attn(X, Dq, Dk):
    with np.errstate(all="ignore"):              # Accelerate BLAS raises spurious FP flags
        S = (X @ Dq) @ (X @ Dk).T
        S -= S.max(axis=1, keepdims=True)
        W = np.exp(S)
    return W / W.sum(axis=1, keepdims=True)


def step(X, Dq, Dk, Vb, Vc, h):
    with np.errstate(all="ignore"):
        F = ((attn(X, Dq, Dk) @ X) @ Vc.T) @ Vb.T
        v = F - np.sum(X * F, axis=1)[:, None] * X
        Y = X + h[:, None] * v
    assert np.isfinite(Y).all()
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def ang(A, B):
    return np.degrees(np.arccos(np.clip(np.sum(A * B, axis=1), -1, 1)))


def nn(X):
    G = X @ X.T
    np.fill_diagonal(G, -np.inf)
    return G.argmax(1)


def main():
    if not os.path.exists(STEPS):
        print("[E11] computing real per-layer step sizes ...", flush=True)
        build_steps()
    z, hz = np.load(CACHE), np.load(STEPS)
    heads = {}
    for name in z.files:
        m, l, h, f = name.split("|")
        heads.setdefault(f"{m}|{l}|{h}", {})[f] = z[name]
    rng = np.random.default_rng(0)
    rows = []
    for key in sorted(heads):
        hd = heads[key]
        Dq, Dk, Vb, Vc = hd["Dq"], hd["Dk"], hd["Vb"], hd["Vc"]
        Sq, Sk = np.hstack([Dq, Dk]) / np.sqrt(2), np.hstack([Dk, Dq]) / np.sqrt(2)
        a2 = 0.5 * (np.sum((Dq.T @ Dq) * (Dk.T @ Dk)) - np.sum((Dq.T @ Dk) * (Dk.T @ Dq)))
        P, Q = rng.normal(size=Dq.shape), rng.normal(size=Dk.shape)
        r2 = 0.5 * (np.sum((P.T @ P) * (Q.T @ Q)) - np.sum((P.T @ Q) * (Q.T @ P)))
        c = np.sqrt(np.sqrt(a2 / r2))
        Rq = np.hstack([Sq, c * P / np.sqrt(2), -c * Q / np.sqrt(2)])
        Rk = np.hstack([Sk, c * Q / np.sqrt(2), c * P / np.sqrt(2)])
        m, l, h = key.split("|")
        for p, X0 in enumerate(hd["starts"]):
            hstep = hz[key][p]
            X = {"real": X0.copy(), "sym": X0.copy(), "rand": X0.copy()}
            fac = {"real": (Dq, Dk), "sym": (Sq, Sk), "rand": (Rq, Rk)}
            for L in range(1, LREC[-1] + 1):
                for k in X:
                    X[k] = step(X[k], *fac[k], Vb, Vc, hstep)
                if L in LREC:
                    moved = ang(X["real"], X0)
                    rec = dict(model=m, layer=int(l), head=int(h), passage=p, L=L, sign=float(hd["sign"]),
                               h_med=float(np.median(hstep)), moved=float(moved.mean()),
                               cos_real=float((np.sum(X["real"] @ X["real"].T) - N) / (N * (N - 1))))
                    for k, (A_, B_) in {"sym": ("real", "sym"), "rand": ("rand", "sym")}.items():
                        d = ang(X[A_], X[B_])
                        rec[f"gap_{k}"] = float(d.mean())
                        rec[f"rel_{k}"] = float(np.linalg.norm(X[A_] - X[B_]) / max(np.linalg.norm(X[A_] - X0), 1e-300))
                        rec[f"nn_{k}"] = float(np.mean(nn(X[A_]) != nn(X[B_])))
                    rows.append(rec)
    with open(os.path.join(OUT, "E11_antisym_depth.json"), "w") as f:
        json.dump(rows, f)

    q = lambda a: f"{np.median(a):6.2f} [{np.percentile(a, 25):.2f}-{np.percentile(a, 75):.2f}]"
    for m in sorted({r["model"] for r in rows}):
        g0 = [r for r in rows if r["model"] == m and r["L"] == 1]
        print(f"\n{m}: real per-layer step h (median over heads) {np.median([r['h_med'] for r in g0]):.3f}")
        print(f"{'L':>3} | {'moved (deg)':>18} | {'real vs sym (deg)':>18} {'relative':>18} {'nn differs':>18} | "
              f"{'random vs sym':>18} {'relative':>18} {'nn differs':>18} | {'mean cos':>8}")
        for L in LREC:
            g = [r for r in rows if r["model"] == m and r["L"] == L]
            print(f"{L:3d} | {q([r['moved'] for r in g])} | {q([r['gap_sym'] for r in g])} {q([r['rel_sym'] for r in g])} "
                  f"{q([r['nn_sym'] for r in g])} | {q([r['gap_rand'] for r in g])} {q([r['rel_rand'] for r in g])} "
                  f"{q([r['nn_rand'] for r in g])} | {np.median([r['cos_real'] for r in g]):8.2f}")
    print("\nsaved results/review/E11_antisym_depth.json")


if __name__ == "__main__":
    main()
