"""E5: the rotating ring and triangle on S^4 (n = 5) (Appendix "Higher dimensions").

Setting 'plane': D_s = diag(0,0,1,1,1), whose zero eigenspace is the (e1,e2) plane, so the resting set
is a ring; A rotates the (e1,e2) plane at rate 1 and the (e3,e4) plane at rate theta2.
Prediction from S^2: ring in the (e1,e2) plane, spinning at eps/2, then a rotating triangle with rate
Omega_3 = 2 s sinh(eps s) / (1 + 2 cosh(eps s)), s = sqrt(3)/2.
Grid: eps in {0.1, 0.3, 0.6}, theta2 in {0, 0.5, 2}, 3 seeds, N = 120.
Setting 'sphere3': D_s = diag(0,0,0,0,1) (resting set is a 3-sphere), A rotates (e1,e2) at rate 1 and
(e3,e4) at rate theta2 in {0.5, 1, 2}; eps = 0.3, 3 seeds.  Exploratory: what is the end state?
Flow: true exponential kernel, V = -Id, projected Euler dt = 0.05; T = 6000 (eps = 0.1) or 3000,
stopping early at 3 tight clusters.  Recorded every 50 time units: off-plane mass L_perp, spin in the
(e1,e2) plane, cluster sizes.  Figure: results/figures/highdim.png.
Run from inside export/:  PYTHONPATH=. python verification/highdim_check.py
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
FIG = os.path.abspath(os.path.join(HERE, os.pardir, "results", "figures", "highdim.png"))
DT, REC, N = 0.05, 50.0, 120
SMOKE = os.environ.get("SMOKE") == "1"
WORKERS = int(os.environ.get("WORKERS", "2"))


def make_A(theta2):
    A = np.zeros((5, 5))
    A[1, 0], A[0, 1] = 1.0, -1.0
    A[3, 2], A[2, 3] = theta2, -theta2
    return A


def field(X, D):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)
    return F - np.sum(X * F, axis=1)[:, None] * X


def clusters(X, tol=0.1):
    reps, members = [], []
    for i, x in enumerate(X):
        for k, r in enumerate(reps):
            if np.dot(x, r) > np.cos(tol):
                members[k].append(i); break
        else:
            reps.append(x); members.append([i])
    spread = 0.0
    for m in members:
        c = X[m].mean(0); c /= np.linalg.norm(c)
        spread = max(spread, float(np.arccos(np.clip(X[m] @ c, -1, 1)).max()))
    return sorted((len(m) for m in members), reverse=True), spread


def omega3(eps):
    s = np.sqrt(3) / 2
    return 2 * s * np.sinh(eps * s) / (1 + 2 * np.cosh(eps * s))


def run(job):
    setting, eps, theta2, seed = job
    Ds = np.diag([0, 0, 1.0, 1.0, 1.0]) if setting == "plane" else np.diag([0, 0, 0, 0, 1.0])
    D = Ds + eps * make_A(theta2)
    X = np.random.default_rng(seed).normal(size=(N, 5))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    T = (6000.0 if eps < 0.2 else 3000.0) if not SMOKE else 100.0
    every, hits, t_conv, hist = int(round(REC / DT)), 0, None, []
    for s in range(1, int(round(T / DT)) + 1):
        v = field(X, D)
        X = X + DT * v
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if s % every == 0:
            r12 = np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 1e-12)
            r34 = np.maximum(X[:, 2] ** 2 + X[:, 3] ** 2, 1e-12)
            sizes, spread = clusters(X)
            hist.append(dict(t=s * DT, L_perp=float(np.mean(np.sum(X[:, 2:] ** 2, 1))),
                             L_e5=float(np.mean(X[:, 4] ** 2)),
                             spin12=float(np.mean((X[:, 0] * v[:, 1] - X[:, 1] * v[:, 0]) / r12)),
                             spin34=float(np.mean((X[:, 2] * v[:, 3] - X[:, 3] * v[:, 2]) / r34)),
                             k=len(sizes), spread=spread))
            hits = hits + 1 if (len(sizes) == 3 and spread < 0.02) else 0
            if hits == 2:
                t_conv = s * DT
                break
    sizes, spread = clusters(X)
    cov = np.sort(np.linalg.eigvalsh(X.T @ X / N))[::-1]
    return dict(setting=setting, eps=eps, theta2=theta2, seed=seed, sizes=sizes, converged=t_conv is not None,
                t_conv=t_conv, cov_eigs=cov.tolist(), hist=hist, final=X.tolist() if seed == 0 else None)


def ring_spin(r):
    """Median spin in the (e1,e2) plane while the off-plane mass is small but before clustering."""
    vals = [h["spin12"] for h in r["hist"] if h["L_perp"] < 1e-3 and h["k"] > 6]
    return float(np.median(vals)) if vals else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    seeds = range(3) if not SMOKE else range(1)
    jobs = [("plane", e, th, s) for e in ([0.1, 0.3, 0.6] if not SMOKE else [0.6]) for th in [0.0, 0.5, 2.0] for s in seeds]
    jobs += [("sphere3", 0.3, th, s) for th in [0.5, 1.0, 2.0] for s in seeds]
    jobs.sort(key=lambda j: -(2 if j[1] < 0.2 else 1))
    t0, out = time.time(), []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            out.append(r)
            print(f"[E5] {i}/{len(jobs)} done ({time.time() - t0:.0f}s): {r['setting']} eps={r['eps']} "
                  f"theta2={r['theta2']} seed={r['seed']}: sizes={r['sizes'][:6]} converged={r['converged']} "
                  f"t_conv={r['t_conv']}", flush=True)
    with open(os.path.join(OUT, "E5_highdim.json"), "w") as f:
        json.dump(out, f)

    print("\nSUMMARY: setting 'plane' (ring expected in the (e1,e2) plane)")
    for r in sorted([r for r in out if r["setting"] == "plane"], key=lambda r: (r["eps"], r["theta2"], r["seed"])):
        last = r["hist"][-1]
        print(f"  eps={r['eps']} theta2={r['theta2']} seed={r['seed']}: final L_perp={last['L_perp']:.1e}  "
              f"ring spin={ring_spin(r):+.4f} (eps/2={r['eps'] / 2:.4f})  final spin={last['spin12']:+.4f} "
              f"(Omega_3={omega3(r['eps']):.4f})  sizes={r['sizes'][:6]}  t_conv={r['t_conv']}")
    print("\nSUMMARY: setting 'sphere3' (resting set is a 3-sphere)")
    for r in sorted([r for r in out if r["setting"] == "sphere3"], key=lambda r: (r["theta2"], r["seed"])):
        last = r["hist"][-1]
        print(f"  theta2={r['theta2']} seed={r['seed']}: final L_e5={last['L_e5']:.1e} spin12={last['spin12']:+.4f} "
              f"spin34={last['spin34']:+.4f} cov eigs={np.round(r['cov_eigs'], 3).tolist()} sizes={r['sizes'][:8]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = next(r for r in out if r["setting"] == "plane" and r["eps"] == (0.1 if not SMOKE else 0.6)
             and r["theta2"] == 0.5 and r["seed"] == 0)
    t = [h["t"] for h in r["hist"]]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].semilogy(t, [h["L_perp"] for h in r["hist"]])
    ax[0].set_xlabel("t"); ax[0].set_title(r"off-plane mass $\langle x_3^2+x_4^2+x_5^2\rangle$")
    ax[1].plot(t, [h["spin12"] for h in r["hist"]], label="measured")
    ax[1].axhline(r["eps"] / 2, ls="--", c="k", lw=0.8, label=r"$\varepsilon/2$ (ring)")
    ax[1].axhline(omega3(r["eps"]), ls=":", c="r", lw=0.8, label=r"$\Omega_3$ (triangle)")
    ax[1].set_xlabel("t"); ax[1].set_title(r"spin in the $(e_1,e_2)$ plane"); ax[1].legend(fontsize=7)
    Xf = np.array(r["final"])
    ax[2].scatter(Xf[:, 0], Xf[:, 1], s=8)
    ax[2].set_aspect("equal"); ax[2].set_xlim(-1.1, 1.1); ax[2].set_ylim(-1.1, 1.1)
    ax[2].set_title(r"final state, projected on $(e_1,e_2)$")
    fig.suptitle(rf"$S^4$: $D_s=\mathrm{{diag}}(0,0,1,1,1)$, $\varepsilon={r['eps']}$, $\theta_2={r['theta2']}$, $N={N}$")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"\nsaved results/review/E5_highdim.json and {os.path.relpath(FIG, os.path.join(HERE, os.pardir))}", flush=True)


if __name__ == "__main__":
    main()
