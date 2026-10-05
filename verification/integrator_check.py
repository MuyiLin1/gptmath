"""E4: does the time-stepper change the results?  Projected forward Euler (dt = 0.05, 0.025) vs
projected RK4 (dt = 0.05; every stage is renormalized onto the sphere).

(a) Growth rates at the uniform rotating ring (eps = 0.3, N = 200, beta = 1, sparse A):
    out-of-plane m=2 (lambda_2 = I_2/I_0) and in-plane m=3 (mu_3, eq. inplane-eigen), fitted as in
    growth_rate_check.py while the seeded amplitude is between 3x and 100x its start.
(b) Time for the ring to coarsen into the rotating triangle (eps = 0.6, N = 60, beta = 1, 10 random
    seeds): first time with 3 clusters, every token within 0.02 rad of its cluster.
Output: results/review/E4_integrator.json and tables.  ~10-20 min on 2 cores.
Run from inside export/:  PYTHONPATH=. python verification/integrator_check.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import numpy as np
from multiprocessing import Pool
from scipy.special import iv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "review"))
A = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
SMOKE = os.environ.get("SMOKE") == "1"
WORKERS = int(os.environ.get("WORKERS", "2"))
SCHEMES = [("euler", 0.05), ("euler", 0.025), ("rk4", 0.05)]


def field(X, D):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)
    return F - np.sum(X * F, axis=1)[:, None] * X


def nrm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def step(X, D, scheme, dt):
    if scheme == "euler":
        return nrm(X + dt * field(X, D))
    k1 = field(X, D)
    k2 = field(nrm(X + 0.5 * dt * k1), D)
    k3 = field(nrm(X + 0.5 * dt * k2), D)
    k4 = field(nrm(X + dt * k3), D)
    return nrm(X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


def lam2(k):
    return iv(2, k) / iv(0, k)


def mu(m, k):
    c = lambda n: iv(n, k) * (-1j) ** n
    gp = lambda q: k / 2 * (c(q - 1) + c(q + 1))
    G = lambda q: (c(q - 1) - c(q + 1)) / (2j)
    Gp = lambda q: 0.5 * (c(q - 1) + c(q + 1)) + k / (4j) * (c(q - 2) - c(q + 2))
    Om = G(0) / c(0)
    return ((Gp(0) - Gp(m) - Om * (gp(0) - gp(m))) / c(0)).real


def growth(job):
    kind, scheme, dt = job
    eps, N, amp = 0.3, 200, 1e-4
    D = np.diag([0, 0, 1.0]) + eps * A
    phi = 2 * np.pi * np.arange(N) / N
    if kind == "oop":
        X = nrm(np.stack([np.cos(phi), np.sin(phi), amp * np.cos(2 * phi)], 1))
    else:
        ph = phi + amp * np.cos(3 * phi)
        X = np.stack([np.cos(ph), np.sin(ph), np.zeros(N)], 1)

    def a(X):
        ph = np.arctan2(X[:, 1], X[:, 0])
        return abs(2 / N * np.sum(X[:, 2] * np.exp(-2j * ph))) if kind == "oop" else abs(np.mean(np.exp(3j * ph)))
    T = 1.5 * np.log(100) / lam2(eps) if not SMOKE else 50
    steps, every = int(round(T / dt)), max(1, int(round(T / dt)) // 2000)
    a0, ts, ys = a(X), [], []
    for s in range(steps + 1):
        if s % every == 0:
            v = a(X); ts.append(s * dt); ys.append(v)
            if v > 100 * a0:
                break
        X = step(X, D, scheme, dt)
    ts, ys = np.array(ts), np.array(ys)
    win = (ys >= 3 * a0) & (ys <= 100 * a0)
    if win.sum() < 5:
        win = np.ones_like(ys, dtype=bool)
    return dict(kind=kind, scheme=scheme, dt=dt, rate=float(np.polyfit(ts[win], np.log(ys[win]), 1)[0]))


def tight3(X):
    reps, members = [], []
    for i, x in enumerate(X):
        for k, r in enumerate(reps):
            if np.dot(x, r) > np.cos(0.1):
                members[k].append(i); break
        else:
            reps.append(x); members.append([i])
    if len(members) != 3:
        return False
    for m in members:
        c = X[m].mean(0); c /= np.linalg.norm(c)
        if np.arccos(np.clip(X[m] @ c, -1, 1)).max() > 0.02:
            return False
    return True


def coarsen(job):
    seed, scheme, dt = job
    D = np.diag([0, 0, 1.0]) + 0.6 * A
    X = nrm(np.random.default_rng(seed).normal(size=(60, 3)))
    T = 3000.0 if not SMOKE else 100.0
    every = int(round(10.0 / dt))
    for s in range(1, int(round(T / dt)) + 1):
        X = step(X, D, scheme, dt)
        if s % every == 0 and tight3(X):
            return dict(seed=seed, scheme=scheme, dt=dt, t_tri=s * dt)
    return dict(seed=seed, scheme=scheme, dt=dt, t_tri=None)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    gjobs = [(k, s, dt) for k in ("oop", "inp") for s, dt in SCHEMES]
    cjobs = [(seed, s, dt) for seed in range(10 if not SMOKE else 1) for s, dt in SCHEMES]
    with Pool(WORKERS) as pool:
        g = []
        for i, r in enumerate(pool.imap_unordered(growth, gjobs), 1):
            g.append(r)
            print(f"[E4] growth {i}/{len(gjobs)} ({time.time() - t0:.0f}s): {r}", flush=True)
        c = []
        for i, r in enumerate(pool.imap_unordered(coarsen, cjobs), 1):
            c.append(r)
            print(f"[E4] coarsening {i}/{len(cjobs)} ({time.time() - t0:.0f}s): {r}", flush=True)
    with open(os.path.join(OUT, "E4_integrator.json"), "w") as f:
        json.dump(dict(growth=g, coarsening=c), f, indent=1)

    print(f"\n(a) growth rates at eps = 0.3 (predicted lambda_2 = {lam2(0.3):.5e}, mu_3 = {mu(3, 0.3):.5e})")
    for s, dt in SCHEMES:
        l = next(r["rate"] for r in g if r["kind"] == "oop" and r["scheme"] == s and r["dt"] == dt)
        m = next(r["rate"] for r in g if r["kind"] == "inp" and r["scheme"] == s and r["dt"] == dt)
        print(f"  {s:5s} dt={dt:<5}: lambda_2 {l:.5e} ({l / lam2(0.3) - 1:+.2%})   mu_3 {m:.5e} ({m / mu(3, 0.3) - 1:+.2%})")
    print("\n(b) time to reach the rotating triangle (eps = 0.6, N = 60)")
    for s, dt in SCHEMES:
        ts = [r["t_tri"] for r in c if r["scheme"] == s and r["dt"] == dt]
        ok = [t for t in ts if t is not None]
        print(f"  {s:5s} dt={dt:<5}: reached in {len(ok)}/{len(ts)} runs; median t = "
              f"{np.median(ok) if ok else float('nan'):.0f}; per seed: {sorted(ts, key=lambda t: (t is None, t or 0))}")
    print("\nsaved results/review/E4_integrator.json", flush=True)


if __name__ == "__main__":
    main()
