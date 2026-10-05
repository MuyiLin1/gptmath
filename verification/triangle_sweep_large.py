"""E3: large sweep of the long-time state (Observation "selection of the rotating triangle").

Grid (random starts): eps in {0.1, 0.3, 0.6, 1.0}, N in {30, 60, 61, 90, 100, 150} (61 and 100 are
not divisible by 3), beta in {0.5, 1, 2}, 10 seeds.  Extra starts at N = 60, beta = 1,
eps in {0.3, 0.6}, 10 seeds each: 'cap' (tokens in a 0.3-rad cap around a random point),
'pole' (0.2-rad cap around e3), 'ring' (equatorial ring with 0.05 noise).
Flow: true exponential kernel, D = diag(0,0,beta) + eps*A_sparse, V = -Id, projected Euler dt = 0.05.
Horizon T = max(3000, 1.5 * 1500 * (0.3/eps)^2) (the ring breaks on times ~ 1/eps^2).
Every 100 time units the tokens are grouped into clusters (greedy, 0.1 rad); a run stops early once it
has 3 tight clusters (every token within 0.02 rad of its cluster) at two consecutive checks.
Recorded: final cluster sizes, whether converged, time of convergence, max |z|.
Output: results/review/E3_triangle_sweep.json and a summary.  ~1-2 h on 2 cores.
Run from inside export/:  PYTHONPATH=. python verification/triangle_sweep_large.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import time
import numpy as np
from collections import Counter
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "review"))
A = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
DT, CHECK = 0.05, 100.0
SMOKE = os.environ.get("SMOKE") == "1"
WORKERS = int(os.environ.get("WORKERS", "2"))


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
                members[k].append(i)
                break
        else:
            reps.append(x); members.append([i])
    spread = 0.0
    for m in members:
        c = X[m].mean(0); c /= np.linalg.norm(c)
        spread = max(spread, float(np.arccos(np.clip(X[m] @ c, -1, 1)).max()))
    return sorted((len(m) for m in members), reverse=True), spread


def horizon(eps):
    return 200.0 if SMOKE else max(3000.0, 1.5 * 1500.0 * (0.3 / eps) ** 2)


def start(kind, N, rng):
    if kind == "random":
        X = rng.normal(size=(N, 3))
    elif kind in ("cap", "pole"):
        c = rng.normal(size=3) if kind == "cap" else np.array([0, 0, 1.0])
        c /= np.linalg.norm(c)
        X = c + (0.3 if kind == "cap" else 0.2) * rng.normal(size=(N, 3)) / np.sqrt(3)
    else:
        ph = rng.uniform(0, 2 * np.pi, N)
        X = np.stack([np.cos(ph), np.sin(ph), np.zeros(N)], 1) + 0.05 * rng.normal(size=(N, 3))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def run(job):
    eps, N, beta, seed, kind = job
    D = np.diag([0, 0, beta]) + eps * A
    X = start(kind, N, np.random.default_rng(seed))
    T, every = horizon(eps), int(round(CHECK / DT))
    steps, hits, t_conv = int(round(T / DT)), 0, None
    for s in range(1, steps + 1):
        X = X + DT * field(X, D)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if s % every == 0:
            sizes, spread = clusters(X)
            hits = hits + 1 if (len(sizes) == 3 and spread < 0.02) else 0
            if hits == 2:
                t_conv = s * DT
                break
    sizes, spread = clusters(X)
    return dict(eps=eps, N=N, beta=beta, seed=seed, start=kind, sizes=sizes, k=len(sizes),
                k_dominant=sum(sz >= 0.1 * N for sz in sizes), spread=spread, converged=t_conv is not None,
                t_conv=t_conv, t_end=s * DT, max_abs_z=float(np.abs(X[:, 2]).max()))


def main():
    os.makedirs(OUT, exist_ok=True)
    eps_list = [0.1, 0.3, 0.6, 1.0] if not SMOKE else [0.6]
    N_list = [30, 60, 61, 90, 100, 150] if not SMOKE else [30, 61]
    betas = [0.5, 1.0, 2.0] if not SMOKE else [1.0]
    seeds = range(10) if not SMOKE else range(1)
    jobs = [(e, n, b, s, "random") for e in eps_list for n in N_list for b in betas for s in seeds]
    for kind in ["cap", "pole", "ring"]:
        for e in ([0.3, 0.6] if not SMOKE else [0.6]):
            jobs += [(e, 60, 1.0, s, kind) for s in seeds]
    jobs.sort(key=lambda j: -horizon(j[0]) * j[1] ** 2)
    t0, out = time.time(), []
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            out.append(r)
            print(f"[E3] {i}/{len(jobs)} done ({(time.time() - t0) / 60:.1f} min): eps={r['eps']} N={r['N']} "
                  f"beta={r['beta']} seed={r['seed']} {r['start']}: sizes={r['sizes'][:6]} "
                  f"{'converged at t=%.0f' % r['t_conv'] if r['converged'] else 'NOT converged by t=%.0f' % r['t_end']}",
                  flush=True)
            if i % 20 == 0:
                with open(os.path.join(OUT, "E3_triangle_sweep.partial.json"), "w") as f:
                    json.dump(out, f)
    with open(os.path.join(OUT, "E3_triangle_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("\nSUMMARY")
    rnd = [r for r in out if r["start"] == "random"]
    print(f"random starts: {len(rnd)} runs; three tight clusters: {sum(r['converged'] for r in rnd)}")
    for e in eps_list:
        rs = [r for r in rnd if r["eps"] == e]
        print(f"  eps={e}: {sum(r['converged'] for r in rs)}/{len(rs)} converged to 3 clusters; "
              f"median t_conv={np.median([r['t_conv'] for r in rs if r['converged']] or [np.nan]):.0f}; "
              f"end states (k_dominant): {dict(Counter(r['k_dominant'] for r in rs))}")
    for n in N_list:
        rs = [r for r in rnd if r["N"] == n and r["converged"]]
        print(f"  N={n}: cluster sizes of converged runs: {dict(Counter(tuple(r['sizes']) for r in rs))}")
    for kind in ["cap", "pole", "ring"]:
        rs = [r for r in out if r["start"] == kind]
        print(f"  start={kind}: {sum(r['converged'] for r in rs)}/{len(rs)} converged; "
              f"end states: {dict(Counter(r['k_dominant'] for r in rs))}")
    bad = [r for r in out if not r["converged"]]
    print(f"not converged: {len(bad)} runs" + ("".join(
        f"\n  eps={r['eps']} N={r['N']} beta={r['beta']} seed={r['seed']} {r['start']} sizes={r['sizes'][:8]}"
        for r in bad[:40])))
    print("\nsaved results/review/E3_triangle_sweep.json", flush=True)


if __name__ == "__main__":
    main()
