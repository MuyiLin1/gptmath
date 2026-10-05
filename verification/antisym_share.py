"""E10: how much of a trained head's token motion comes from the antisymmetric part of its attention matrix?
(Appendix "Trained heads as dynamical systems")

For every head of GPT-2 small and BERT-base (matrices and real token states from trained_head_flow.py,
cache results/review/E7_heads.npz) and both text passages (N = 64 tokens), at the real token states X:
    v      = P_X( sum_j a_ij(D) V X_j )        velocity with the true D = D_s + eps A
    v_s    = same with D replaced by D_s       (antisymmetric part removed)
    share  = |v - v_s| / |v|                   (Frobenius over tokens)
    shift  = mean_i  (1/2) sum_j |a_ij(D) - a_ij(D_s)|   (attention mass moved by the antisymmetric part)
    angle  = mean_i angle(v_i, v_s,i)
for V = the head's own value-output map, V = -Id and V = +Id.
Baseline: the same with the antisymmetric part replaced by a random antisymmetric matrix of equal
Frobenius norm (rank 2 d_h, from Gaussian factors of the head's shape), to separate "large because
attention is sharp" from the head's actual antisymmetric part.
Run from inside export/ after trained_head_flow.py has built its cache:
    PYTHONPATH=. python verification/antisym_share.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "review"))
CACHE = os.path.join(OUT, "E7_heads.npz")


def load():
    z = np.load(CACHE)
    heads = {}
    for name in z.files:
        m, l, h, f = name.split("|")
        heads.setdefault((m, int(l), int(h)), {})[f] = z[name]
    return heads


def attn(X, Dq, Dk):
    with np.errstate(all="ignore"):              # Accelerate BLAS raises spurious FP flags
        S = (X @ Dq) @ (X @ Dk).T
        S -= S.max(axis=1, keepdims=True)
        W = np.exp(S)
    return W / W.sum(axis=1, keepdims=True)


def vel(X, W, Vmap):
    with np.errstate(all="ignore"):
        F = Vmap(W @ X)
        v = F - np.sum(X * F, axis=1)[:, None] * X
    assert np.isfinite(v).all()
    return v


def compare(X, Dq, Dk, Sq, Sk, Vmap):
    W, Ws = attn(X, Dq, Dk), attn(X, Sq, Sk)
    v, vs = vel(X, W, Vmap), vel(X, Ws, Vmap)
    nv, nvs = np.linalg.norm(v, axis=1), np.linalg.norm(vs, axis=1)
    cos = np.sum(v * vs, axis=1) / np.maximum(nv * nvs, 1e-300)
    return dict(share=float(np.linalg.norm(v - vs) / np.linalg.norm(v)),
                shift=float(0.5 * np.abs(W - Ws).sum(axis=1).mean()),
                angle=float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()))


def main():
    heads = load()
    rng = np.random.default_rng(0)
    rows = []
    for (m, l, h), hd in sorted(heads.items()):
        Dq, Dk, Vb, Vc = hd["Dq"], hd["Dk"], hd["Vb"], hd["Vc"]
        Sq, Sk = np.hstack([Dq, Dk]) / np.sqrt(2), np.hstack([Dk, Dq]) / np.sqrt(2)   # D_s = Sq Sk^T
        # antisymmetric part eps A = (Dq Dk^T - Dk Dq^T)/2, its Frobenius norm via the small Gram matrices
        a2 = 0.5 * (np.sum((Dq.T @ Dq) * (Dk.T @ Dk)) - np.sum((Dq.T @ Dk) * (Dk.T @ Dq)))
        P, Q = rng.normal(size=Dq.shape), rng.normal(size=Dk.shape)
        r2 = 0.5 * (np.sum((P.T @ P) * (Q.T @ Q)) - np.sum((P.T @ Q) * (Q.T @ P)))
        c = np.sqrt(np.sqrt(a2 / r2))                                                # factors enter squared
        Rq = np.hstack([Sq, c * P / np.sqrt(2), -c * Q / np.sqrt(2)])
        Rk = np.hstack([Sk, c * Q / np.sqrt(2), c * P / np.sqrt(2)])                # D_s + c^2 (P Q^T - Q P^T)/2
        maps = {"own": lambda Y: (Y @ Vc.T) @ Vb.T, "-Id": lambda Y: -Y, "+Id": lambda Y: Y}
        for p, X in enumerate(hd["starts"]):
            for vn, Vmap in maps.items():
                r = compare(X, Dq, Dk, Sq, Sk, Vmap)
                rb = compare(X, Rq, Rk, Sq, Sk, Vmap)
                rows.append(dict(model=m, layer=l, head=h, passage=p, V=vn, sign=float(hd["sign"]),
                                 **r, **{k + "_rand": v for k, v in rb.items()}))
    with open(os.path.join(OUT, "E10_antisym_share.json"), "w") as f:
        json.dump(rows, f)

    print("median over heads and passages (middle half in brackets)")
    q = lambda a: f"{np.median(a):.2f} [{np.percentile(a, 25):.2f}-{np.percentile(a, 75):.2f}]"
    for m in sorted({r["model"] for r in rows}):
        for vn in ("own", "-Id", "+Id"):
            g = [r for r in rows if r["model"] == m and r["V"] == vn]
            sh = np.array([r["share"] for r in g])
            print(f"{m:18} V={vn:4} | share {q(sh)}  >0.5: {np.mean(sh > 0.5):.2f} | "
                  f"angle {q([r['angle'] for r in g])} deg | attn moved {q([r['shift'] for r in g])} | "
                  f"random A: share {q([r['share_rand'] for r in g])}, attn moved {q([r['shift_rand'] for r in g])}")
        g = [r for r in rows if r["model"] == m and r["V"] == "own"]
        print(f"{'':18} per-layer median share (own V): " + " ".join(
            f"{np.median([r['share'] for r in g if r['layer'] == L]):.2f}" for L in range(12)))
    print("saved results/review/E10_antisym_share.json")


if __name__ == "__main__":
    main()
