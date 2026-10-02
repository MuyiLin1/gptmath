"""Zero vertical force on a flat ring (Prop. "drift", Prop. "aligned"; replaces the
old "~1e-18 residual" claim, code reviewer point #2).

On the uniform equatorial ring with D_s = diag(0,0,beta), the first-order field
V_1 = -P_x Cov(x^T A y, y) has NO component along e3, for ANY antisymmetric A
(any dual axis omega). The paper proves this; here we check it two ways:
  (1) the closed-form V_1 evaluated on the ring;
  (2) the true (full-exponential) field: (V(eps) - V(-eps)) / (2 eps), whose
      e3-part must vanish to machine precision as well.
Deterministic.  Run from inside export/:  PYTHONPATH=. python verification/ring_normal_force_check.py
"""
import numpy as np


def true_field(X, D):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)                                   # V = -Id
    return F - np.sum(X * F, axis=1)[:, None] * X


def V1_closed_form(X):
    """V_1(x) = -P_x Cov(x^T A y, y) with uniform weights on the ring (D_s y . x = 0)."""
    def f(A):
        G = X @ A @ X.T                              # G[i,j] = x_i^T A y_j
        m_gy = (G[:, :, None] * X[None, :, :]).mean(1)
        m_g = G.mean(1)[:, None]
        m_y = X.mean(0)[None, :]
        C = -(m_gy - m_g * m_y)
        return C - np.sum(X * C, axis=1)[:, None] * X
    return f


def cross_matrix(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0.0]])


rng = np.random.default_rng(0)
print(f"{'N':>5} {'beta':>5} {'omega':>28} {'max|e3.V1| closed':>18} {'max|e3.V1| true-field FD':>25} {'drift c (meas)':>15} {'c = w3/2':>9}")
for N in (48, 200, 1000):
    phi = 2 * np.pi * np.arange(N) / N
    X = np.stack([np.cos(phi), np.sin(phi), np.zeros(N)], 1)
    for beta in (0.5, 2.0):
        for trial in range(2):
            w = rng.normal(size=3)
            A = cross_matrix(w)                     # A x = w x x
            V1 = V1_closed_form(X)(A)
            eps = 1e-4
            Ds = np.diag([0, 0, beta])
            fd = (true_field(X, Ds + eps * A) - true_field(X, Ds - eps * A)) / (2 * eps)
            tang = np.stack([-np.sin(phi), np.cos(phi), np.zeros(N)], 1)
            c_meas = np.mean(np.sum(V1 * tang, 1))
            print(f"{N:>5} {beta:>5} {np.array2string(w, precision=3):>28} "
                  f"{np.abs(V1[:, 2]).max():>18.1e} {np.abs(fd[:, 2]).max():>25.1e} "
                  f"{c_meas:>15.6f} {w[2]/2:>9.6f}")
print("\nExpected: both e3 columns at machine precision (<~1e-12; the finite-difference")
print("column is limited by the step 1e-4); drift c matches w3/2 (Prop. 'drift').")
