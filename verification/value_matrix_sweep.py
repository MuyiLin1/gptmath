"""E1: long-time behavior under different value matrices (Section "role of the value matrix",
Appendix "Simulations with other value matrices", Remark "Effect of the value matrix").

Grid: D_s in {diag(0,0,1), Id}, V in {-Id, +Id, -D, +D}, eps in {0.1, 0.3, 0.6}, 5 seeds.
Starts: random (both D_s) and, for D_s = diag(0,0,1), the uniform equatorial ring plus 1e-3 noise.
N = 200, dt = 0.05, T = 400, true exponential flow, projected Euler.
Recorded at the end (speed and spin also averaged over the last 50 time units):
  speed   mean |velocity|
  spin    mean angular velocity about e3
  R1, R3  |<e^{i phi}>|, |<e^{3 i phi}>| (unevenness around the ring)
  zrms    sqrt(<z^2>)
  k       number of clusters (greedy, 0.1 rad)
Output: results/review/E1_value_matrices.json and a summary table (median over seeds).
Run from inside export/:  PYTHONPATH=. python verification/value_matrix_sweep.py
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
N, DT, T, TAVG = 200, 0.05, 400.0, 50.0
SMOKE = os.environ.get("SMOKE") == "1"
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


def run(job):
    ds_name, v_name, eps, seed, start = job
    Ds = np.diag([0, 0, 1.0]) if ds_name == "axis" else np.eye(3)
    D = Ds + eps * A
    V = {"-Id": -np.eye(3), "+Id": np.eye(3), "-D": -D, "+D": D}[v_name]
    rng = np.random.default_rng(seed)
    if start == "random":
        X = rng.normal(size=(N, 3))
    else:
        ph = 2 * np.pi * np.arange(N) / N
        X = np.stack([np.cos(ph), np.sin(ph), np.zeros(N)], 1) + 1e-3 * rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    steps = int(round((T if not SMOKE else 20) / DT))
    navg = int(round(TAVG / DT)) if not SMOKE else 10
    sp_acc, spin_acc = [], []
    for s in range(steps):
        v = field(X, D, V)
        if s >= steps - navg:
            rho2 = np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 1e-12)
            sp_acc.append(np.mean(np.linalg.norm(v, axis=1)))
            spin_acc.append(np.mean((X[:, 0] * v[:, 1] - X[:, 1] * v[:, 0]) / rho2))
        X = X + DT * v
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    v = field(X, D, V)
    rho2 = np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 1e-12)
    phi = np.arctan2(X[:, 1], X[:, 0])
    return dict(Ds=ds_name, V=v_name, eps=eps, seed=seed, start=start,
                speed=float(np.mean(np.linalg.norm(v, axis=1))), speed_avg=float(np.mean(sp_acc)),
                spin=float(np.mean((X[:, 0] * v[:, 1] - X[:, 1] * v[:, 0]) / rho2)), spin_avg=float(np.mean(spin_acc)),
                R1=float(abs(np.mean(np.exp(1j * phi)))), R3=float(abs(np.mean(np.exp(3j * phi)))),
                zrms=float(np.sqrt(np.mean(X[:, 2] ** 2))), k=n_clusters(X))


def main():
    os.makedirs(OUT, exist_ok=True)
    eps_list = [0.1, 0.3, 0.6] if not SMOKE else [0.3]
    seeds = range(5) if not SMOKE else range(1)
    jobs = []
    for ds in ["axis", "iso"]:
        for v in ["-Id", "+Id", "-D", "+D"]:
            for eps in eps_list:
                for seed in seeds:
                    jobs.append((ds, v, eps, seed, "random"))
                    if ds == "axis":
                        jobs.append((ds, v, eps, seed, "ring"))
    t0, out = time.time(), []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            out.append(r)
            print(f"[E1] {i}/{len(jobs)} done ({time.time() - t0:.0f}s): {r['Ds']} V={r['V']} eps={r['eps']} "
                  f"seed={r['seed']} {r['start']}: speed={r['speed_avg']:.2e} spin={r['spin_avg']:+.4f} k={r['k']}",
                  flush=True)
    with open(os.path.join(OUT, "E1_value_matrices.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\nSUMMARY (median over seeds; speed/spin averaged over the last 50 time units)")
    hdr = f"{'D_s':5} {'V':4} {'eps':>4} {'start':6} | {'speed':>9} {'spin':>9} {'eps/2':>6} | {'R1':>6} {'R3':>6} {'zrms':>7} {'k':>4}"
    print(hdr); print("-" * len(hdr))
    keys = sorted({(r["Ds"], r["V"], r["eps"], r["start"]) for r in out})
    for key in keys:
        rs = [r for r in out if (r["Ds"], r["V"], r["eps"], r["start"]) == key]
        med = lambda f: float(np.median([r[f] for r in rs]))
        print(f"{key[0]:5} {key[1]:4} {key[2]:>4} {key[3]:6} | {med('speed_avg'):9.2e} {med('spin_avg'):+9.4f} "
              f"{key[2] / 2:6.3f} | {med('R1'):6.3f} {med('R3'):6.3f} {med('zrms'):7.1e} {med('k'):4.0f}")
    print("\nsaved results/review/E1_value_matrices.json", flush=True)


if __name__ == "__main__":
    main()
