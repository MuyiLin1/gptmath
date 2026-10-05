"""E3b: is the three-cluster state the long-time state, not just the first one reached?
(Observation "triangle selection", Appendix "The triangle sweep", Remark "Unequal triangles")

D_s = diag(0,0,1), sparse A, V = -Id, beta = 1, random starts, N in {60, 61}, eps in {0.1, 0.3, 0.6, 1.0},
5 seeds, horizon T = max(8000, 2250 (0.3/eps)^2 * 4).  Two variants: noiseless, and with a kick of size
1e-8 (independent Gaussian per coordinate, then renormalized) every 10 time units, so that tokens of a
cluster never become bit-identical and slow internal instabilities can act.
Every 50 time units: cluster sizes (0.1 rad), number of dominant clusters (>= 10% of tokens), the largest
angular radius of a cluster.  Reported: first time at 3 tight clusters, fraction of the last half of the
run spent with exactly 3 dominant clusters, number of break-ups (3 tight -> not tight), final sizes.
Run from inside export/:  PYTHONPATH=. python verification/triangle_longtime.py
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
A = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
DT, REC, KICK, TIGHT = 0.05, 50.0, 1e-8, 0.02
WORKERS = int(os.environ.get("WORKERS", "2"))
SMOKE = os.environ.get("SMOKE") == "1"


def clusters(X, tol=0.1):
    reps, mem = [], []
    for i, x in enumerate(X):
        for k, r in enumerate(reps):
            if x @ r > np.cos(tol):
                mem[k].append(i); break
        else:
            reps.append(x); mem.append([i])
    rad = 0.0
    for m in mem:
        c = X[m].mean(0); c /= np.linalg.norm(c)
        rad = max(rad, float(np.arccos(np.clip(X[m] @ c, -1, 1)).max()))
    return sorted((len(m) for m in mem), reverse=True), rad


def run(job):
    N, eps, seed, noisy = job
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    D = np.diag([0, 0, 1.0]) + eps * A
    T = max(8000.0, 4 * 2250 * (0.3 / eps) ** 2) if not SMOKE else 200.0
    krng = np.random.default_rng(1000 + seed)
    every, kick = int(round(REC / DT)), int(round(10 / DT))
    hist, t_first, breakups, was_tight = [], None, 0, False
    for s in range(1, int(round(T / DT)) + 1):
        S = X @ D @ X.T
        S -= S.max(1, keepdims=True)
        W = np.exp(S); W /= W.sum(1, keepdims=True)
        F = -(W @ X)
        F -= (F * X).sum(1, keepdims=True) * X
        X = X + DT * F
        if noisy and s % kick == 0:
            X = X + KICK * krng.normal(size=X.shape)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if s % every == 0:
            sizes, rad = clusters(X)
            dom = sum(n >= 0.1 * N for n in sizes)
            tight = len(sizes) == 3 and rad < TIGHT
            if tight and t_first is None:
                t_first = s * DT
            if was_tight and not tight:
                breakups += 1
            was_tight = tight
            hist.append((s * DT, sizes[:6], dom, rad))
    late = [h for h in hist if h[0] >= T / 2]
    return dict(N=N, eps=eps, seed=seed, noisy=noisy, T=T, t_first=t_first, breakups=breakups,
                frac3_late=float(np.mean([h[2] == 3 for h in late])),
                frac_tight_late=float(np.mean([len(h[1]) == 3 and h[3] < TIGHT for h in late])),
                final_sizes=hist[-1][1], final_rad=hist[-1][3], hist=hist)


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(N, eps, seed, noisy) for eps in (0.1, 0.3, 0.6, 1.0) for N in (60, 61)
            for seed in range(5 if not SMOKE else 1) for noisy in (False, True)]
    jobs.sort(key=lambda j: j[1])                  # longest (small eps) first
    t0, out = time.time(), []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            out.append(r)
            print(f"[E3b] {i}/{len(jobs)} ({time.time() - t0:.0f}s) N={r['N']} eps={r['eps']} seed={r['seed']} "
                  f"noisy={int(r['noisy'])}: t_first={r['t_first']} breakups={r['breakups']} "
                  f"3-dominant(late)={r['frac3_late']:.2f} tight(late)={r['frac_tight_late']:.2f} "
                  f"final={r['final_sizes']} rad={r['final_rad']:.1e}", flush=True)
    with open(os.path.join(OUT, "E3b_triangle_longtime.json"), "w") as f:
        json.dump(out, f)
    print("\nSUMMARY")
    for key in sorted({(r["eps"], r["N"], r["noisy"]) for r in out}):
        rs = [r for r in out if (r["eps"], r["N"], r["noisy"]) == key]
        print(f"eps={key[0]} N={key[1]} noisy={int(key[2])}: 3-dominant late {np.mean([r['frac3_late'] for r in rs]):.2f}, "
              f"tight late {np.mean([r['frac_tight_late'] for r in rs]):.2f}, breakups {[r['breakups'] for r in rs]}, "
              f"final {[tuple(r['final_sizes'][:3]) for r in rs]}")
    print("saved results/review/E3b_triangle_longtime.json", flush=True)


if __name__ == "__main__":
    main()
