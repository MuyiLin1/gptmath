"""Rotation rate of the regular k-gon cluster states (Prop. "kgon").

Paper formula (D_s = diag(0,0,beta), V = -Id, dual axis omega = e3, theta_r = 2 pi r/k):
    Omega_k = - sum_r exp(-eps sin theta_r) sin theta_r / sum_r exp(-eps sin theta_r)
so Omega_2 = 0 and Omega_k = eps/2 + O(eps^3) for k >= 3.
We compare it with the tangential velocity of the true field at a vertex.
Deterministic, < 1 s.  Run from inside export/:  PYTHONPATH=. python verification/kgon_rate_check.py
"""
import numpy as np

A = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])


def field(X, D):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)
    return F - np.sum(X * F, axis=1)[:, None] * X


print(f"{'k':>3} {'eps':>5} {'beta':>5} {'measured':>12} {'formula':>12} {'eps/2':>8}")
for k in (2, 3, 4, 5, 6):
    for eps in (0.1, 0.6):
        for beta in (1.0, 3.0):
            th = 2 * np.pi * np.arange(k) / k
            X = np.repeat(np.stack([np.cos(th), np.sin(th), 0 * th], 1), 10, axis=0)  # k groups of 10
            v = field(X, np.diag([0, 0, beta]) + eps * A)
            meas = X[0, 0] * v[0, 1] - X[0, 1] * v[0, 0]
            w = np.exp(-eps * np.sin(th))
            form = -np.sum(w * np.sin(th)) / np.sum(w)
            print(f"{k:>3} {eps:>5} {beta:>5} {meas:>+12.8f} {form:>+12.8f} {eps/2:>8.4f}")
print("\nExpected: measured == formula to machine precision; independent of beta; 0 for k = 2.")
