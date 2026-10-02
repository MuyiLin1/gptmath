"""Tilt of the skewed ring: direction, size, and the height correlation
(Cor. "geometry", proof of Prop. "skew"; code reviewer point #1).

Skewed A (dual vector omega = (3,0,3)), D_s = diag(0,0,1), V = -Id, N = 400,
dt = 0.05, 8000 steps (T = 400) -- the setup of tilt_scaling.py.

For each run we report
  * the ring normal nu (smallest-eigenvalue eigenvector of the second moment);
  * tilt angle from e3;
  * cos of the angle between the horizontal part of the tilt and omega x e3
    (= -y): +1 means the normal moved exactly toward omega x e3;
  * corr(z_i, e3.(omega x X_i)): the correlation quoted in the paper.
Two kinds of start: a random cloud on the sphere, and a ring near the equator.
Expected (paper): direction toward omega x e3 in every run; at eps = 0.1 tilt ~10 deg
from a random start vs ~1.7 deg from a near-ring start; correlation ~0.8-0.9 from
random starts.  Runtime: a few minutes.
Run from inside export/:  PYTHONPATH=. python verification/tilt_check.py
"""
import numpy as np

A = np.array([[0., -3, 0], [3, 0, -3], [0, 3, 0]])
OMEGA = np.array([3., 0, 3])
TARGET = np.cross(OMEGA, [0, 0, 1.0]); TARGET /= np.linalg.norm(TARGET)   # omega x e3 direction


def flow(X, D):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = W @ X
    return -(F - np.sum(X * F, axis=1)[:, None] * X)


def start(kind, seed, N):
    rng = np.random.default_rng(seed)
    if kind == "random":
        X = rng.normal(size=(N, 3))
    else:                                            # near-ring start
        ph = 2 * np.pi * rng.random(N)
        X = np.column_stack([np.cos(ph), np.sin(ph), 0.02 * rng.normal(size=N)])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def settle(eps, kind, seed, N=400, dt=0.05, steps=8000):
    X = start(kind, seed, N)
    D = np.diag([0, 0, 1.0]) + eps * A
    for _ in range(steps):
        X = X + dt * flow(X, D)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def stats(X):
    w, V = np.linalg.eigh(X.T @ X / len(X))
    nu = V[:, 0] if V[2, 0] > 0 else -V[:, 0]
    tilt = np.degrees(np.arccos(np.clip(nu[2], -1, 1)))
    h = nu[:2] / max(np.linalg.norm(nu[:2]), 1e-15)
    cosdir = h @ TARGET[:2]
    corr = np.corrcoef(X[:, 2], np.cross(OMEGA, X)[:, 2])[0, 1]
    return nu, tilt, cosdir, corr


print(f"omega x e3 direction = {TARGET}")
print(f"{'eps':>5} {'start':>9} {'seed':>4} {'normal nu':>28} {'tilt(deg)':>9} {'cos(dir, w x e3)':>16} {'corr':>6}")
for eps in (0.04, 0.10):
    for kind in ("random", "near-ring"):
        for seed in (0, 1, 2):
            nu, tilt, cosdir, corr = stats(settle(eps, kind, seed))
            print(f"{eps:>5} {kind:>9} {seed:>4} {np.array2string(nu, precision=4):>28} "
                  f"{tilt:>9.2f} {cosdir:>16.3f} {corr:>6.3f}")
