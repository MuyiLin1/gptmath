"""Zero-sum height ripples on the rotating ring: slow growth (the buckle) and
possible leveling-off (Remark "L-scope", Cor. "no-threshold"; issue 3).

Paper convention: D = diag(0,0,beta) + eps*A, sparse A (omega = e3), so kappa = eps.
Near the ring, for an m=2 ripple z_i = a cos(2 phi_i), the paper's results give
    dL/dt  ~  2*lambda_2*L - 2*beta*L^2,     lambda_2 = I_2(kappa)/I_0(kappa) ~ kappa^2/8,
so small ripples grow (the buckle) and large ones shrink, suggesting a leveling-off at
    L*  ~  lambda_2 / beta.
Part 1 checks the formula for dL/dt at several amplitudes (same idea as T3 in
lyapunov_test.py, but in the paper's convention).
Part 2 runs the flow from a small and a large ripple and prints L(t): if both approach
the same value near L*, the leveling-off is real (at this N, over this horizon).
Deterministic (uniform ring).  Runtime ~1 min.
Run from inside export/:  PYTHONPATH=. python verification/ripple_buckle_check.py
"""
import numpy as np
from scipy.special import iv

N, BETA, EPS = 60, 1.0, 0.2
A = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
D = np.diag([0, 0, BETA]) + EPS * A
phi = 2 * np.pi * np.arange(N) / N
RING = np.stack([np.cos(phi), np.sin(phi), np.zeros(N)], 1)
lam2 = iv(2, EPS) / iv(0, EPS)
Lstar = lam2 / BETA


def field(X):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)
    return F - np.sum(X * F, axis=1)[:, None] * X


def ripple(a):
    X = RING + a * np.cos(2 * phi)[:, None] * np.array([0, 0, 1.0])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def L(X):
    return np.mean(X[:, 2] ** 2)


def Ldot(X):
    return 2 * np.mean(X[:, 2] * field(X)[:, 2])


print(f"N={N} beta={BETA} eps=kappa={EPS}  lambda_2={lam2:.5f}  predicted leveling-off L*={Lstar:.5f}\n")
print("Part 1: dL/dt vs prediction 2*lambda_2*L - 2*beta*L^2")
print(f"{'amp':>7} {'L':>11} {'dL/dt':>12} {'predicted':>12}")
for a in (0.3, 0.15, 0.08, 0.04, 0.02, 0.01):
    X = ripple(a)
    print(f"{a:>7} {L(X):>11.3e} {Ldot(X):>+12.3e} {2*lam2*L(X) - 2*BETA*L(X)**2:>+12.3e}")

print("\nPart 2: L(t) from a small and a large ripple (dt = 0.05)")
dt, steps = 0.05, 30000
runs = {"small (a=0.005)": ripple(0.005), "large (a=0.3)": ripple(0.3)}
hist = {k: [] for k in runs}
for s in range(steps + 1):
    for k in runs:
        if s % 3000 == 0:
            hist[k].append(L(runs[k]))
        X = runs[k]
        X = X + dt * field(X)
        runs[k] = X / np.linalg.norm(X, axis=1, keepdims=True)
print(f"{'t':>7} " + " ".join(f"{k:>18}" for k in runs))
for i in range(len(hist[next(iter(runs))])):
    print(f"{i*3000*dt:>7.0f} " + " ".join(f"{hist[k][i]:>18.3e}" for k in runs))
print(f"\nCompare the late values with L* = {Lstar:.3e}.  Also watch for in-plane clumping (m=3),")
print("which grows faster (rate 3 kappa^2/16) and can take over at long times.")
