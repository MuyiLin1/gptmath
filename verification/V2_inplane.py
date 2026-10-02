"""
Second-order field, in-plane effects (Remark "Second order"): what does V_2 actually PREDICT?  The IN-PLANE second-order
dynamics -- concretely, the order-eps^2 correction to the ring's drift speed and
the order-eps^2 modulation of its density.

On the uniform ring measure mu (a great circle with normal nu), expand the exact
mean-field drift speed about the rotation axis:
    Omega(eps) = <Vf_eps . that> / R   =   eps*Om1 + eps^2*Om2 + eps^3*Om3 + ...
with that = unit tangent.  The coefficients are the tangential averages of the
velocity-field expansion:
    Om1 = mean_phi  V1(x(phi)) . that(phi)
    Om2 = mean_phi  V2(x(phi)) . that(phi)
We verify Om1,Om2 (from the closed forms) reproduce the exact Omega(eps), i.e.
V_2 is exactly the eps^2 piece of the in-plane drift.

We also report the order-eps^2 DENSITY signal: the tangential divergence of V2
around the ring, whose harmonics say where tokens bunch up at second order.
"""
import numpy as np


def Px(x, v):
    return v - (v @ x) * x


def exact_drift_field(x, Y, D):
    s = Y @ (D.T @ x); s -= s.max()
    w = np.exp(s); w /= w.sum()
    return Px(x, (w[:, None] * Y).sum(0))


def V1_closed(x, Y, Ds, A):
    s = Y @ (Ds @ x); s -= s.max(); a = np.exp(s); a /= a.sum()
    g = Y @ (A.T @ x); dg = g - a @ g; dy = Y - a @ Y
    return Px(x, (a[:, None] * (dg[:, None] * dy)).sum(0))


def V2_closed(x, Y, Ds, A):
    s = Y @ (Ds @ x); s -= s.max(); a = np.exp(s); a /= a.sum()
    g = Y @ (A.T @ x); dg = g - a @ g; dy = Y - a @ Y
    return 0.5 * Px(x, (a[:, None] * (dg[:, None] ** 2 * dy)).sum(0))


def dual_axis(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]])


def ring_basis(nu):
    a = np.array([1., 0, 0]) if abs(nu[0]) < 0.9 else np.array([0., 1, 0])
    e1 = a - (a @ nu) * nu; e1 /= np.linalg.norm(e1)
    return e1, np.cross(nu, e1)


def harmonics(sig, phi, mmax=4):
    return np.array([2 * np.hypot((sig * np.cos(m * phi)).mean(),
                                  (sig * np.sin(m * phi)).mean())
                     for m in range(1, mmax + 1)])


def analyze(name, A, Ds, M=720):
    nu = dual_axis(A); nu = nu / np.linalg.norm(nu)
    e1, e2 = ring_basis(nu)
    phi = 2 * np.pi * np.arange(M) / M
    ring = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    that = -np.sin(phi)[:, None] * e1 + np.cos(phi)[:, None] * e2

    # closed-form tangential averages -> drift coefficients
    v1t = np.array([V1_closed(ring[k], ring, Ds, A) @ that[k] for k in range(M)])
    v2t = np.array([V2_closed(ring[k], ring, Ds, A) @ that[k] for k in range(M)])
    Om1, Om2 = v1t.mean(), v2t.mean()

    # exact drift speed at several eps
    print(f"\n=== {name} ===  ring normal nu={np.round(nu,3)}")
    print(f"  drift coeffs from fields:  Om1(eps^1)={Om1:+.4f}   Om2(eps^2)={Om2:+.4f}")
    print(f"  {'eps':>5} {'Omega_exact':>12} {'eps*Om1':>10} {'+eps^2*Om2':>12} {'resid':>10}")
    for eps in [0.05, 0.1, 0.2, 0.3]:
        D = Ds + eps * A
        ome = np.array([exact_drift_field(ring[k], ring, D) @ that[k] for k in range(M)]).mean()
        pred1 = eps * Om1
        pred2 = eps * Om1 + eps ** 2 * Om2
        print(f"  {eps:5.2f} {ome:12.5f} {pred1:10.5f} {pred2:12.5f} {ome-pred2:10.2e}")

    # order-eps^2 density signal: tangential divergence of V2 around the ring
    v2_full = np.array([V2_closed(ring[k], ring, Ds, A) for k in range(M)])
    v2_tan = (v2_full * that).sum(1)
    ddphi = np.gradient(v2_tan, phi)            # d/dphi of tangential V2 ~ -density rate
    hdens = harmonics(ddphi - ddphi.mean(), phi)
    print(f"  order-eps^2 density harmonics (div V2) m=1..4 = {np.round(hdens,4)}"
          f"  -> dominant m={int(np.argmax(hdens))+1}")
    return Om1, Om2


if __name__ == "__main__":
    # A_sparse: dual axis -e3 (the transpose of the paper's A_sparse; only the sense of
    # rotation differs, so Om1 = -1/2 here). A_skewed: dual vector (1,0,1), i.e. the
    # paper's A_skewed / 3.
    A_sparse = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 0]])
    A_skewed = np.array([[0., -1, 0], [1, 0, -1], [0, 1, 0]])
    Ds = np.diag([0., 0., 1.])
    print("What V_2 predicts IN-PLANE: the eps^2 drift correction and density modulation.")
    print("(Om2 ~ 0 + tiny residual => drift is odd in eps for the symmetric ring;")
    print(" the genuine eps^2 V_2 effect is the density modulation harmonic.)")
    analyze("aligned (A_sparse)", A_sparse, Ds)
    analyze("skewed  (A_skewed)", A_skewed, Ds)
    print("\nDONE")
