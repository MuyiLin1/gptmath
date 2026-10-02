"""Check: does the non-gradient obstruction (Theorem "V_1 is non-gradient") extend
from Ds=0 to the experiment's Ds=diag(0,0,1) in closed form?

Paper conventions: V = -Id, so V1[mu](x) = -P_x Cov^0(x^T A y, y), and A x = omega x x.

Claim: yes. On the equatorial great circle gamma (perp to e3) the height z = x_3 = 0,
so the symmetric kernel e^{x.Ds y} = e^{z_x z_y} = 1 there; the softmax weights a^0
are uniform in y; and V1[sigma](x) collapses to the same rotation field as in the
isotropic Ds=0 case.  Hence the circulation around gamma (oriented counterclockwise
about +e3) is exactly (2*pi/n) (omega . e3), INDEPENDENT of whether Ds=0 or
Ds=diag(0,0,1).

We verify numerically that circulation(Ds=diag(0,0,1)) = circulation(Ds=0)
= (2*pi/3) (omega . e3) to high accuracy, for both A_sparse (omega = e3, expect
+2*pi/3) and A_skewed (omega = (3,0,3), expect +2*pi).
"""
import numpy as np

n = 3
TWO_PI_OVER_N = 2 * np.pi / n

A_SPARSE = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])      # paper's A_sparse, omega = e3
A_SKEWED = np.array([[0., -3, 0], [3, 0, -3], [0, 3, 0]])     # paper's A_skewed, omega = (3,0,3)


def sphere_samples(m, seed=0):
    """approx-uniform points on S^2 via a Fibonacci sphere."""
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]


def V1_at(x, A, Ds, Y):
    """First-order field V1[sigma](x) = -P_x Cov^0(x.A y, y), sigma ~ uniform,
    weights a^0(x,y) ∝ exp(x . Ds y)."""
    w = np.exp(Y @ Ds @ x)          # unnormalised softmax weights e^{x.Ds y}
    w = w / w.sum()
    g = Y @ (A.T @ x)               # g(x,y) = x^T A y  for each y
    ybar = w @ Y                    # <y>^0
    gbar = w @ g                    # <g>^0
    gy = (w[:, None] * (g[:, None] * Y)).sum(0)   # <g y>^0
    cov = -(gy - gbar * ybar)       # minus sign from V = -Id
    return cov - (cov @ x) * x      # tangential projection P_x^perp


def circulation(A, Ds, M=720, Ny=20000):
    Y = sphere_samples(Ny)
    phi = 2 * np.pi * np.arange(M) / M
    total = 0.0
    dl = 2 * np.pi / M
    for p in phi:
        x = np.array([np.cos(p), np.sin(p), 0.0])       # point on the equator
        t = np.array([-np.sin(p), np.cos(p), 0.0])      # unit tangent
        total += V1_at(x, A, Ds, Y) @ t * dl
    return total


def main():
    Ds0 = np.zeros((3, 3))
    Ds1 = np.diag([0., 0., 1.])
    print("analytic prediction  (2*pi/n) (omega . e3)\n")
    for name, A in [("A_sparse", A_SPARSE), ("A_skewed", A_SKEWED)]:
        omega = np.array([A[2, 1], A[0, 2], A[1, 0]])  # A x = omega x x
        c0 = circulation(A, Ds0)
        c1 = circulation(A, Ds1)
        print(f"{name}:  predicted = {TWO_PI_OVER_N * omega[2]:+.6f}   circ(Ds=0) = {c0:+.6f}"
              f"   circ(Ds=diag(0,0,1)) = {c1:+.6f}   |diff| = {abs(c0 - c1):.2e}")


if __name__ == "__main__":
    main()
