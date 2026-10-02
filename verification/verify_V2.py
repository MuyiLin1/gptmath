"""
Verify the analytic second-order velocity field V_2 (Theorem "V2", Appendix) against the exact
mean-field attention field, by an even/odd finite difference in eps.

Field (V=Id):  Vf_eps[mu](x) = P_x( sum_j w_j y_j ),  w_j ~ exp(x . (Ds+eps A) y_j).
Expansion:     Vf_eps = V0 + eps V1 + eps^2 V2 + O(eps^3).

Because V1 is ODD in eps (linear in g=x.Ay) and V2 is EVEN, the symmetric second
difference isolates V2 with NO contamination from V1 or V3:
    V2_num = ( Vf_{+eps} + Vf_{-eps} - 2 V0 ) / (2 eps^2) + O(eps^2).

Closed forms under the symmetric softmax average  <f>^0 = sum_j a0_j f_j,
a0_j ~ exp(x . Ds y_j),  g_j = x . A y_j,  dg = g-<g>^0,  dy = y-<y>^0:
    V1_cf = P_x <dg dy>^0
    V2_cf = 1/2 P_x <dg^2 dy>^0
"""
import numpy as np

np.random.seed(0)


def Px(x, M):                      # project rows of M onto tangent at x (single x)
    return M - (M @ x) * x


def exact_field(x, Y, D):
    s = Y @ (D.T @ x)              # x . D y_j = (D^T x).y_j
    s -= s.max()
    w = np.exp(s); w /= w.sum()
    return Px(x, (w[:, None] * Y).sum(0))


def V1_closed(x, Y, Ds, A):
    s = Y @ (Ds @ x); s -= s.max()
    a = np.exp(s); a /= a.sum()
    g = Y @ (A.T @ x)             # g_j = x.A y_j
    gm = a @ g
    ym = a @ Y
    dg = g - gm
    dy = Y - ym
    cov = (a[:, None] * (dg[:, None] * dy)).sum(0)   # <dg dy>
    return Px(x, cov)


def V2_closed(x, Y, Ds, A):
    s = Y @ (Ds @ x); s -= s.max()
    a = np.exp(s); a /= a.sum()
    g = Y @ (A.T @ x)
    gm = a @ g
    ym = a @ Y
    dg = g - gm
    dy = Y - ym
    coskew = (a[:, None] * (dg[:, None] ** 2 * dy)).sum(0)  # <dg^2 dy>
    return 0.5 * Px(x, coskew)


def randsphere(N, n):
    Y = np.random.randn(N, n)
    return Y / np.linalg.norm(Y, axis=1, keepdims=True)


def check(label, Ds, A, N=600, n=3, eps=1e-2, ntest=40):
    Y = randsphere(N, n)
    Xt = randsphere(ntest, n)
    e1 = e2 = 0.0
    for x in Xt:
        Dp = Ds + eps * A
        Dm = Ds - eps * A
        V0 = exact_field(x, Y, Ds)
        Vp = exact_field(x, Y, Dp)
        Vm = exact_field(x, Y, Dm)
        V1_num = (Vp - Vm) / (2 * eps)
        V2_num = (Vp + Vm - 2 * V0) / (2 * eps ** 2)
        V1_cf = V1_closed(x, Y, Ds, A)
        V2_cf = V2_closed(x, Y, Ds, A)
        e1 = max(e1, np.linalg.norm(V1_num - V1_cf) / (np.linalg.norm(V1_cf) + 1e-12))
        e2 = max(e2, np.linalg.norm(V2_num - V2_cf) / (np.linalg.norm(V2_cf) + 1e-12))
    print(f"{label:28s}  max rel err  V1={e1:.2e}   V2={e2:.2e}")


if __name__ == "__main__":
    # test matrices: A_sparse has dual axis -e3 (the transpose of the paper's A_sparse),
    # A_skewed is the paper's A_skewed / 3; any antisymmetric A works for this check.
    A_sparse = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 0]])
    A_skewed = np.array([[0., -1, 0], [1, 0, -1], [0, 1, 0]])
    Ds_rep = np.diag([0., 0., 1.])         # repulsive (equator ring) case
    Ds_zero = np.zeros((3, 3))             # isotropic case
    Ds_rand = np.random.randn(3, 3); Ds_rand = 0.5 * (Ds_rand + Ds_rand.T)
    A_rand = np.random.randn(3, 3); A_rand = 0.5 * (A_rand - A_rand.T)

    print("Numerical check of closed-form V1 (known) and V2 (new), eps=1e-2:")
    print("(expected: both errors ~ O(eps^2) ~ 1e-4, the finite-difference truncation)\n")
    check("Ds=diag(0,0,1), A_sparse", Ds_rep, A_sparse)
    check("Ds=diag(0,0,1), A_skewed", Ds_rep, A_skewed)
    check("Ds=0,           A_sparse", Ds_zero, A_sparse)
    check("Ds=random sym,  A_random", Ds_rand, A_rand)

    # tighten eps to confirm second-order convergence of the error
    print("\nConvergence of V2 error as eps shrinks (should fall ~ eps^2):")
    Y = randsphere(600, 3); x = randsphere(1, 3)[0]
    for eps in [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]:
        Vp = exact_field(x, Y, Ds_rep + eps * A_skewed)
        Vm = exact_field(x, Y, Ds_rep - eps * A_skewed)
        V0 = exact_field(x, Y, Ds_rep)
        V2_num = (Vp + Vm - 2 * V0) / (2 * eps ** 2)
        V2_cf = V2_closed(x, Y, Ds_rep, A_skewed)
        err = np.linalg.norm(V2_num - V2_cf) / np.linalg.norm(V2_cf)
        print(f"  eps={eps:7.4f}   rel err={err:.3e}")
    print("\nDONE")
