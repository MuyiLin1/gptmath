"""V = -Id (the paper's model) versus V = -D (Burger et al.'s gradient-flow choice)
in the ring setup (Remark "Effect of the value matrix").

D_s = diag(0,0,1), sparse A (omega = e3), N = 300, random start, dt = 0.05, 4000 steps.
Reports the mean spin rate about e3, the height L = <z^2>, and how unevenly the tokens
sit around the ring (|mean of exp(i phi)|: 0 = even ring, 1 = one clump).
Expected (paper): V = -Id spins at eps/2; V = -D spins ~100x slower, opposite sign,
and the ring is not evenly filled.  Runtime ~1-2 min.
Run from inside export/:  PYTHONPATH=. python verification/value_matrix_compare.py
"""
import numpy as np

A = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])


def vel(X, D, V):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = W @ (X @ V.T)
    return F - np.sum(X * F, axis=1)[:, None] * X


def run(eps, which, N=300, dt=0.05, steps=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    D = np.diag([0, 0, 1.0]) + eps * A
    V = -np.eye(3) if which == "-Id" else -D
    for _ in range(steps):
        X = X + dt * vel(X, D, V)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    v = vel(X, D, V)
    rho2 = np.maximum(X[:, 0] ** 2 + X[:, 1] ** 2, 1e-12)
    spin = np.mean((X[:, 0] * v[:, 1] - X[:, 1] * v[:, 0]) / rho2)
    phi = np.arctan2(X[:, 1], X[:, 0])
    return spin, np.mean(X[:, 2] ** 2), abs(np.mean(np.exp(1j * phi)))


print(f"{'V':>4} {'eps':>5} {'spin rate':>11} {'eps/2':>7} {'height L':>9} {'unevenness':>11}")
for which in ("-Id", "-D"):
    for eps in (0.0, 0.05, 0.1):
        s, L, r = run(eps, which)
        print(f"{which:>4} {eps:>5.2f} {s:>+11.5f} {eps/2:>7.3f} {L:>9.1e} {r:>11.2f}")
