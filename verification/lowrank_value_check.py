"""E8: does the rotation need repulsion in every direction?  (Section "role of the value matrix",
Appendix "Simulations with other value matrices", paragraph "Partial and mixed value matrices")

S^2, D_s = diag(0,0,1), sparse A (omega = e3), V = diag(-1, -1, c):
  c = -1    V = -Id (baseline)
  c = -0.5  weaker push off the ring's plane
  c =  0    Test A: push only within the ring's plane (rank-2 V)
  c = +0.5  mild pull off the plane
  c = +1    Test B: push within the plane, pull along the axis
eps in {0.1, 0.3, 0.6}, 5 seeds, random start and ring start (uniform equator + 1e-3 noise),
N = 200, dt = 0.05, T = 400, true exponential flow, projected Euler.
S^4 (Test D): D_s = diag(0,0,1,1,1), A rotates (e1,e2) at rate 1 and (e3,e4) at rate 0.5,
V in {-Id, -P12 (rank 2 of 5), diag(-1,-1,1,1,1)}, eps in {0.1, 0.3, 0.6}, 3 seeds, N = 120, T = 600.
Recorded (speed and spin averaged over the last 50 time units): speed, spin in the (e1,e2) plane,
off-plane mass L = <|x_perp|^2>, number of clusters (0.1 rad).
EXT=1: threshold check c in {-0.1, +0.1} (S^2, T = 400), and the neutral cases c = 0 (S^2) and
-P12 (S^4) from random starts to T = 4000, with L recorded every 250 time units.
Run from inside export/:  PYTHONPATH=. python verification/lowrank_value_check.py
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
DT, TAVG = 0.05, 50.0
SMOKE = os.environ.get("SMOKE") == "1"
EXT = os.environ.get("EXT") == "1"
WORKERS = int(os.environ.get("WORKERS", "2"))


def field(X, D, V):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = W @ (X @ V.T)
    return F - np.sum(X * F, axis=1)[:, None] * X


def n_clusters(X, tol=0.1):
    reps = []
    for x in X:
        if not any(np.dot(x, r) > np.cos(tol) for r in reps):
            reps.append(x)
    return len(reps)


def setup(job):
    dim, vname, eps = job[0], job[1], job[2]
    if dim == 3:
        D = np.diag([0, 0, 1.0]) + eps * np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
        V = np.diag([-1.0, -1.0, float(vname)])
        return D, V, 200, 400.0
    A = np.zeros((5, 5)); A[1, 0], A[0, 1], A[3, 2], A[2, 3] = 1, -1, 0.5, -0.5
    D = np.diag([0, 0, 1.0, 1, 1]) + eps * A
    V = {"-Id": -np.eye(5), "-P12": -np.diag([1.0, 1, 0, 0, 0]), "mixed": np.diag([-1.0, -1, 1, 1, 1])}[vname]
    return D, V, 120, 600.0


def run(job):
    dim, vname, eps, seed, start = job[:5]
    D, V, N, T = setup(job)
    if len(job) > 5:
        T = job[5]
    rng = np.random.default_rng(seed)
    if start == "random":
        X = rng.normal(size=(N, dim))
    else:
        ph = 2 * np.pi * np.arange(N) / N
        X = np.zeros((N, dim)); X[:, 0], X[:, 1] = np.cos(ph), np.sin(ph)
        X += 1e-3 * rng.normal(size=(N, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    T = T if not SMOKE else 20.0
    steps, navg = int(round(T / DT)), int(round((TAVG if not SMOKE else 5.0) / DT))
    sp, spin, Lhist = [], [], []
    for s in range(steps):
        v = field(X, D, V)
        if s % int(round(250 / DT)) == 0:
            Lhist.append(float(np.mean(np.sum(X[:, 2:] ** 2, axis=1))))
        if s >= steps - navg:
            r2 = np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 1e-12)
            sp.append(np.mean(np.linalg.norm(v, axis=1)))
            spin.append(np.mean((X[:, 0] * v[:, 1] - X[:, 1] * v[:, 0]) / r2))
        X = X + DT * v
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return dict(dim=dim, V=vname, eps=eps, seed=seed, start=start, speed=float(np.mean(sp)),
                spin=float(np.mean(spin)), L=float(np.mean(np.sum(X[:, 2:] ** 2, axis=1))), k=n_clusters(X),
                T=T, L_hist=Lhist)


def main():
    os.makedirs(OUT, exist_ok=True)
    eps_list = [0.1, 0.3, 0.6] if not SMOKE else [0.3]
    jobs = []
    if EXT:
        for eps in eps_list:
            for seed in range(5):
                for c in ["-0.1", "0.1"]:
                    for start in ["random", "ring"]:
                        jobs.append((3, c, eps, seed, start))
                jobs.append((3, "0", eps, seed, "random", 4000.0))
            for seed in range(3):
                jobs.append((5, "-P12", eps, seed, "random", 4000.0))
    for c in (["-1", "-0.5", "0", "0.5", "1"] if not EXT else []):
        for eps in eps_list:
            for seed in (range(5) if not SMOKE else range(1)):
                for start in ["random", "ring"]:
                    jobs.append((3, c, eps, seed, start))
    for vn in (["-Id", "-P12", "mixed"] if not EXT else []):
        for eps in eps_list:
            for seed in (range(3) if not SMOKE else range(1)):
                jobs.append((5, vn, eps, seed, "random"))
    t0, out = time.time(), []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            out.append(r)
            print(f"[E8] {i}/{len(jobs)} ({time.time() - t0:.0f}s) S^{r['dim'] - 1} V={r['V']} eps={r['eps']} "
                  f"seed={r['seed']} {r['start']}: speed={r['speed']:.2e} spin={r['spin']:+.4f} L={r['L']:.1e} k={r['k']}",
                  flush=True)
    with open(os.path.join(OUT, "E8b_lowrank_ext.json" if EXT else "E8_lowrank_value.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nSUMMARY (median over seeds)")
    print(f"{'sph':4} {'V':6} {'eps':>4} {'start':6} | {'speed':>9} {'spin':>9} {'eps/2':>6} | {'L':>8} {'k':>4}")
    for key in sorted({(r["dim"], r["V"], r["eps"], r["start"]) for r in out}):
        rs = [r for r in out if (r["dim"], r["V"], r["eps"], r["start"]) == key]
        med = lambda f: float(np.median([r[f] for r in rs]))
        print(f"S^{key[0] - 1:<2} {key[1]:6} {key[2]:>4} {key[3]:6} | {med('speed'):9.2e} {med('spin'):+9.4f} "
              f"{key[2] / 2:6.3f} | {med('L'):8.1e} {med('k'):4.0f}  (k per seed: {sorted(r['k'] for r in rs)})")
        if rs[0]["T"] > 1000:
            print("      L every 250:", " ".join(f"{x:.1e}" for x in np.median([r["L_hist"] for r in rs], axis=0)))
    print("\nsaved", "E8b_lowrank_ext.json" if EXT else "E8_lowrank_value.json", flush=True)


if __name__ == "__main__":
    main()
