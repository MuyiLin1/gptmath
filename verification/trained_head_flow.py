"""E7: run the mean-field flow with the weights of real trained heads (Section "role of the value matrix",
Appendix "Trained heads as dynamical systems").

For every head h of GPT-2 small and BERT-base (layer-norm gain folded in, biases dropped):
    D_h = d * G W_Q,h W_K,h^T G / sqrt(d_h)   (score between unit vectors u, u' is u^T D_h u'; d = 768)
    V_h = (W_V,h W_O,h)^T G-folded, rescaled to spectral norm 1 (a change of time unit only)
in column convention, so that dX_i/dt = P_{X_i}( sum_j a_ij V_h X_j ),  a_ij = softmax_j(X_i^T D_h X_j).
Starts: N = 64 consecutive tokens of real text, their hidden states at the input of the head's layer,
centered and normalized onto the unit sphere (two passages).  No causal mask (the model of the paper).
Controls per head: D_h or its symmetric part, times three value matrices: V_h itself, its symmetric part,
and its pushing part (the negative eigenspace of the symmetric part, "push").
Integrator: projected Heun with step-size control (local error 1e-6), T = 1000.
Recorded every 10 time units: mean speed, mean pairwise cosine, number of clusters (0.1 rad); at the end,
the top real part of the spectrum of V and the alignment of the tokens with its eigenspace.
Head class from E2: push if tr(V_h) < 0.
Run from inside export/ in an env with torch + transformers + safetensors (weights read from the local
HF cache; set HF_HUB_OFFLINE=1 if the hub is unreachable):
    PYTHONPATH=. python verification/trained_head_flow.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "review"))
CACHE = os.path.join(OUT, "E7_heads.npz")
N, T, TOL = 64, 1000.0, 1e-6
TREC = [0, 1, 5] + list(range(10, 1001, 10))
SMOKE = os.environ.get("SMOKE") == "1"
WORKERS = int(os.environ.get("WORKERS", "4"))
ONLY = os.environ.get("ONLY")
MODEL = os.environ.get("MODEL")
OUTNAME = os.environ.get("OUTNAME", "E7_trained_heads")
PASSAGES = [
    "The history of the city begins in the early middle ages, when a small group of fishermen built their "
    "houses on the islands of the lagoon. Over the following centuries the settlement grew into a powerful "
    "republic whose ships carried spices, silk and grain across the sea, and whose merchants kept careful "
    "records of every voyage, every contract and every debt that they owed or were owed.",
    "To prepare the bread, mix the flour with the salt and the yeast in a large bowl, then add the warm water "
    "slowly while stirring with a wooden spoon. When the dough comes together, turn it out onto a floured "
    "table and knead it for about ten minutes, until it is smooth and elastic. Leave it to rise in a warm "
    "place for an hour, then shape it into a loaf and bake it until the crust is golden.",
]


def build():
    """Extract (Dq, Dk, Vb, Vc, starts) per head; D = Dq Dk^T and V = Vb Vc are low-rank factorizations."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from transformers import AutoModel, AutoTokenizer
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
            assert ids.shape[1] >= N + 2, ids.shape
            lo = 0 if model == "gpt2" else 1                         # skip [CLS]
            hs.append([h[0, lo:lo + N].double().numpy() for h in net(ids, output_hidden_states=True).hidden_states])
        for l in range(12):
            if model == "gpt2":
                W = sd[f"h.{l}.attn.c_attn.weight"]
                Wq, Wk, Wv = W[:, :d], W[:, d:2 * d], W[:, 2 * d:]
                Wo, g, b = sd[f"h.{l}.attn.c_proj.weight"], sd[f"h.{l}.ln_1.weight"], None
            else:
                pre = "embeddings.LayerNorm" if l == 0 else f"encoder.layer.{l - 1}.output.LayerNorm"
                g = sd.get(f"{pre}.weight", sd.get(f"{pre}.gamma"))
                b = sd.get(f"{pre}.bias", sd.get(f"{pre}.beta"))
                a = f"encoder.layer.{l}.attention"
                Wq, Wk, Wv = (sd[f"{a}.self.{n}.weight"].T for n in ("query", "key", "value"))
                Wo = sd[f"{a}.output.dense.weight"].T
            starts = []
            for P in hs:
                x = P[l]
                u = x - x.mean(1, keepdims=True) if b is None else (x - b) / g
                starts.append(u / np.linalg.norm(u, axis=1, keepdims=True))
            for h in range(H):
                sl = slice(h * dh, (h + 1) * dh)
                Dq = np.sqrt(d) * g[:, None] * Wq[:, sl] / dh ** 0.25
                Dk = np.sqrt(d) * g[:, None] * Wk[:, sl] / dh ** 0.25
                Vb, Vc = Wo[sl].T, (g[:, None] * Wv[:, sl]).T          # V = Vb @ Vc  (d x d, column convention)
                Vb = Vb / np.linalg.norm(Vb @ Vc, 2)
                tr = float(np.trace(Vc @ Vb))
                key = f"{model}|{l}|{h}"
                out[key] = dict(Dq=Dq, Dk=Dk, Vb=Vb, Vc=Vc, starts=np.stack(starts), trace=tr,
                                sign=float(tr / np.abs(np.linalg.eigvals(Vc @ Vb)).sum()))
    flat = {}
    for k, v in out.items():
        for f, a in v.items():
            flat[f"{k}|{f}"] = np.asarray(a)
    np.savez(CACHE, **flat)


def load():
    z = np.load(CACHE)
    heads = {}
    for name in z.files:
        m, l, h, f = name.split("|")
        heads.setdefault(f"{m}|{l}|{h}", {})[f] = z[name]
    return heads


def factors(hd, symD, vmode):
    Dq, Dk, Vb, Vc = hd["Dq"], hd["Dk"], hd["Vb"], hd["Vc"]
    if symD:                                   # (Dq Dk^T + Dk Dq^T)/2
        Dq, Dk = np.hstack([Dq, Dk]) / np.sqrt(2), np.hstack([Dk, Dq]) / np.sqrt(2)
    if vmode in ("sym", "push"):               # (Vb Vc + Vc^T Vb^T)/2
        Vb, Vc = np.hstack([Vb, Vc.T]) / 2, np.vstack([Vc, Vb.T])
    if vmode == "push":                        # keep only the negative eigenvalues of the symmetric part
        q, r = np.linalg.qr(Vb)
        lam, w = np.linalg.eigh(r @ Vc @ q)
        lam, w = np.real(lam), q @ w
        keep = lam < 0
        Vb, Vc = w[:, keep] * lam[keep], w[:, keep].T
    return Dq, Dk, Vb, Vc


def field(X, Dq, Dk, Vb, Vc):
    with np.errstate(all="ignore"):              # Accelerate BLAS raises spurious FP flags; checked below
        S = (X @ Dq) @ (X @ Dk).T
        S -= S.max(axis=1, keepdims=True)
        W = np.exp(S)
        W /= W.sum(axis=1, keepdims=True)
        F = (W @ (X @ Vc.T)) @ Vb.T
        v = F - np.sum(X * F, axis=1)[:, None] * X
    if not np.isfinite(v).all():
        raise FloatingPointError("non-finite velocity")
    return v


def n_clusters(X, tol=0.1):
    reps = []
    for x in X:
        if not any(np.dot(x, r) > np.cos(tol) for r in reps):
            reps.append(x)
    return len(reps)


def stats(X, v):
    with np.errstate(all="ignore"):
        G = X @ X.T
    return dict(speed=float(np.mean(np.linalg.norm(v, axis=1))),
                cos=float((G.sum() - N) / (N * (N - 1))), k=n_clusters(X))


def top_eigspace(Vb, Vc):
    """Max real part of the nonzero eigenvalues of V = Vb Vc and an orthonormal basis of its (real) eigenspace."""
    lam, w = np.linalg.eig(Vc @ Vb)
    i = int(np.argmax(lam.real))
    u = Vb @ w[:, i]
    B = np.stack([u.real, u.imag], 1) if abs(lam[i].imag) > 1e-9 else u.real[:, None]
    q, _ = np.linalg.qr(B)
    return float(lam[i].real), float(abs(lam[i].imag)), q


def run(job):
    key, symD, vmode, p = job
    hd = HEADS[key]
    f = factors(hd, symD, vmode)
    lam_re, lam_im, Q = top_eigspace(f[2], f[3])
    X = hd["starts"][p].copy()
    t, dt, rec, nsteps = 0.0, 1e-3, [], 0
    targets = [x for x in TREC if x <= (T if not SMOKE else 10)]
    v = field(X, *f)
    rec.append(dict(t=0.0, **stats(X, v)))
    i = 1
    while i < len(targets):
        dt = min(dt, targets[i] - t)
        X1 = X + dt * v
        X1 /= np.linalg.norm(X1, axis=1, keepdims=True)
        v1 = field(X1, *f)
        X2 = X + 0.5 * dt * (v + v1)
        X2 /= np.linalg.norm(X2, axis=1, keepdims=True)
        err = float(np.max(np.linalg.norm(X2 - X1, axis=1)))
        if err <= TOL or dt < 1e-7:
            X, t, nsteps = X2, t + dt, nsteps + 1
            v = field(X, *f)
            if abs(t - targets[i]) < 1e-12:
                rec.append(dict(t=float(t), **stats(X, v)))
                i += 1
        dt *= min(4.0, max(0.2, 0.9 * np.sqrt(TOL / max(err, 1e-300))))
    m, l, h = key.split("|")
    Vx = (X @ f[3].T) @ f[2].T
    return dict(model=m, layer=int(l), head=int(h), symD=symD, vmode=vmode, passage=p,
                trace=float(hd["trace"]), sign=float(hd["sign"]), steps=nsteps, rec=rec,
                lam_top=lam_re, lam_top_im=lam_im,
                align_top=float(np.median(np.linalg.norm(X @ Q, axis=1))),
                range_frac=float(np.median(np.linalg.norm(Vx, axis=1))))


def init(heads):
    global HEADS
    HEADS = heads


def summarize(path):
    """Tables for the paper: per model, head class (push: tr V < 0) and condition."""
    rs = json.load(open(path))
    late = lambda r: float(np.median([q["speed"] for q in r["rec"] if q["t"] >= 0.8 * r["rec"][-1]["t"]]))
    print(f"{len(rs)} runs, {len({(r['model'], r['layer'], r['head']) for r in rs})} heads")
    print(f"{'model':5} {'class':5} {'V':5} {'D':5} | {'n':>4} {'1 cluster':>9} {'k med':>6} {'align':>6} "
          f"{'lam_top>0':>9} | {'moving':>6} {'speed med':>9}")
    for m in sorted({r["model"] for r in rs}):
        for cls in ("push", "pull"):
            for vm in ("full", "sym", "push"):
                for sd in (False, True):
                    g = [r for r in rs if r["model"] == m and (r["sign"] < 0) == (cls == "push")
                         and r["vmode"] == vm and r["symD"] == sd]
                    if not g:
                        continue
                    sp = np.array([late(r) for r in g])
                    print(f"{m[:5]:5} {cls:5} {vm:5} {'sym' if sd else 'full':5} | {len(g):4d} "
                          f"{np.mean([r['rec'][-1]['k'] == 1 for r in g]):9.2f} {np.median([r['rec'][-1]['k'] for r in g]):6.0f} "
                          f"{np.median([r['align_top'] for r in g]):6.2f} {np.mean([r['lam_top'] > 0 for r in g]):9.2f} | "
                          f"{np.mean(sp > 1e-4):6.2f} {np.median(sp):9.1e}")
    print("\npaired effect of the antisymmetric part of D (same head, V, start): late speed full D / symmetric D")
    idx = {(r["model"], r["layer"], r["head"], r["vmode"], r["passage"], r["symD"]): r for r in rs}
    for m in sorted({r["model"] for r in rs}):
        for vm in ("full", "sym", "push"):
            for cls in ("push", "pull"):
                q = [late(idx[k[:5] + (False,)]) / max(late(idx[k[:5] + (True,)]), 1e-12)
                     for k in idx if k[0] == m and k[3] == vm and not k[5] and k[:5] + (True,) in idx
                     and (idx[k]["sign"] < 0) == (cls == "push")]
                if q:
                    q = np.array(q)
                    print(f"  {m[:5]:5} V={vm:5} {cls:5}: n={len(q)}  median ratio {np.median(q):.2f}  "
                          f"full D faster in {np.mean(q > 1):.2f}  >10x faster in {np.mean(q > 10):.2f}")


def main():
    if os.environ.get("SUMMARY"):
        summarize(os.environ["SUMMARY"])
        return
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(CACHE):
        print("[E7] extracting head matrices and hidden states ...", flush=True)
        build()
    heads = load()
    keys = sorted(heads, key=lambda k: (k.split("|")[0], int(k.split("|")[1]), int(k.split("|")[2])))
    if ONLY:
        keys = [k for k in keys if k in ONLY.split(",")]
    if MODEL:
        keys = [k for k in keys if k.startswith(MODEL + "|")]
    jobs = [(k, sd, vm, p) for k in keys for sd in (False, True) for vm in ("full", "sym", "push") for p in (0, 1)]
    if SMOKE:
        jobs = jobs[:8]
    t0, out = time.time(), []
    with Pool(WORKERS, initializer=init, initargs=(heads,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            out.append(r)
            e = r["rec"][-1]
            print(f"[E7] {i}/{len(jobs)} ({time.time() - t0:.0f}s) {r['model']} L{r['layer']}H{r['head']} "
                  f"sign={r['sign']:+.2f} symD={int(r['symD'])} V={r['vmode']} p={r['passage']} "
                  f"steps={r['steps']}: speed0={r['rec'][0]['speed']:.2e} speedT={e['speed']:.2e} "
                  f"cos={e['cos']:+.2f} k={e['k']} lam_top={r['lam_top']:+.2f} align={r['align_top']:.2f}", flush=True)
            if i % 50 == 0:
                with open(os.path.join(OUT, OUTNAME + ".partial.json"), "w") as fh:
                    json.dump(out, fh)
    with open(os.path.join(OUT, OUTNAME + ".json"), "w") as fh:
        json.dump(out, fh)
    print(f"\nsaved results/review/{OUTNAME}.json", flush=True)


if __name__ == "__main__":
    main()
