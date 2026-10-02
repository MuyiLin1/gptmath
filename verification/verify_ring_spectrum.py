"""Spectrum of the rotating ring (Appendix "Spectrum of the rotating ring").

Checks the closed-form in-plane growth rates mu_m(kappa) of the uniform rotating
ring against a brute-force finite-difference Jacobian of the true flow, and
confirms: only the m=3 mode is unstable, mu_3/kappa^2 -> 3/16 at small kappa,
the spectrum is independent of beta and N, mu_3/lambda_2 -> 3/2, and the
nonlinear flow grows at the predicted m=3 rate.  Prints PASS/FAIL per test.
"""
import numpy as np
from scipy.special import iv

# ----------------------------------------------------------------------
# The dynamics (used ONLY for the brute-force checks; no theory here)
# ----------------------------------------------------------------------
def flow(X, D):
    """Right-hand side of the particle ODE with V = -Id."""
    S = X @ D @ X.T
    S = S - S.max(axis=1, keepdims=True)          # softmax stabilisation
    W = np.exp(S); W /= W.sum(axis=1, keepdims=True)
    F = W @ X
    return -(F - np.sum(X * F, axis=1)[:, None] * X)

def A_from_axis(w):
    """Antisymmetric matrix with A x = w cross x."""
    w = np.asarray(w, float)
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

# ----------------------------------------------------------------------
# Brute-force in-plane Jacobian (finite differences, theory-free)
# ----------------------------------------------------------------------
def inplane_jacobian(N, beta, eps, omega=(0, 0, 1), h=1e-6):
    phi = 2 * np.pi * np.arange(N) / N
    D = np.diag([0, 0, beta]) + eps * A_from_axis(omega)
    def azimuthal_velocity(u):
        ph = phi + u
        X = np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], 1)
        V = flow(X, D)
        tang = np.stack([-np.sin(ph), np.cos(ph), np.zeros_like(ph)], 1)
        return np.sum(V * tang, axis=1)
    J = np.zeros((N, N))
    for j in range(N):
        up = np.zeros(N); up[j] = h
        um = np.zeros(N); um[j] = -h
        J[:, j] = (azimuthal_velocity(up) - azimuthal_velocity(um)) / (2 * h)
    return J

def jacobian_mode_eigenvalue(J, m):
    N = J.shape[0]
    v = np.exp(1j * m * 2 * np.pi * np.arange(N) / N)
    return (np.conj(v) @ (J @ v)) / (np.conj(v) @ v)

# ----------------------------------------------------------------------
# The closed form from the theorem
# ----------------------------------------------------------------------
def _c(n, k):
    """Jacobi-Anger coefficient c_n = I_n(k) (-i)^n."""
    return iv(abs(n), k) * ((-1j) ** n)

def mu_closed(m, kappa):
    """Closed-form in-plane eigenvalue for azimuthal wavenumber m."""
    k = kappa
    g0  = _c(0, k)
    gp  = lambda mm: (k / 2) * (_c(mm - 1, k) + _c(mm + 1, k))
    Gf  = lambda mm: (1 / (2j)) * (_c(mm - 1, k) - _c(mm + 1, k))
    Gp  = lambda mm: 0.5 * (_c(mm - 1, k) + _c(mm + 1, k)) \
                     + (k / (4j)) * (_c(mm - 2, k) - _c(mm + 2, k))
    Omega = Gf(0) / g0
    return ((Gp(0) - Gp(m)) - Omega * (gp(0) - gp(m))) / g0

def lambda_oop(q, kappa):
    """Out-of-plane eigenvalue (Discovery 1), for comparison."""
    return -((-1j) ** q) * iv(abs(q), kappa) / iv(0, kappa)

# ----------------------------------------------------------------------
def banner(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)
results = {}

banner("TEST 1  closed form vs brute-force Jacobian  [CENTRAL TEST]")
ok = True
for (N, beta, eps) in [(36, 1.0, 0.3), (48, 1.0, 0.2), (40, 2.0, 0.5)]:
    J = inplane_jacobian(N, beta, eps)
    print(f"  N={N} beta={beta} eps={eps}")
    for m in [1, 2, 3, 4, 5]:
        direct = jacobian_mode_eigenvalue(J, m).real
        closed = mu_closed(m, eps).real
        good = abs(direct - closed) < 1e-5
        ok &= good
        print(f"    m={m}: brute={direct:+.6f}  closed={closed:+.6f}  "
              f"{'ok' if good else 'MISMATCH'}")
results["TEST 1"] = ok

banner("TEST 2  only m=3 is unstable")
kappa = 0.3
mus = {m: mu_closed(m, kappa).real for m in range(8)}
for m, v in mus.items():
    print(f"  m={m}: Re(mu)={v:+.6f}")
unstable = [m for m, v in mus.items() if v > 1e-6]
ok = (unstable == [3]) and (abs(mus[1] + 0.5) < 0.05)
print(f"  unstable modes: {unstable}   (expect exactly [3])")
print(f"  mu_1 = {mus[1]:.4f}  (expect ~ -0.5)")
results["TEST 2"] = ok

banner("TEST 3  small-kappa coefficient equals 3/16 = 0.1875")
for kappa in [0.2, 0.1, 0.05, 0.02, 0.01]:
    ratio = mu_closed(3, kappa).real / kappa ** 2
    print(f"  kappa={kappa:5}: mu_3/kappa^2 = {ratio:.6f}")
ok = abs(mu_closed(3, 0.01).real / 0.01 ** 2 - 3 / 16) < 0.01 * (3 / 16)
print(f"  limit vs 3/16={3/16:.6f}: {'ok' if ok else 'MISMATCH'}")
results["TEST 3"] = ok

banner("TEST 4  beta-independence")
specs = []
for beta in [0.5, 1.0, 2.0, 5.0]:
    J = inplane_jacobian(32, beta, 0.3)
    specs.append(np.sort(np.linalg.eigvals(J).real))
    print(f"  beta={beta}: top eigenvalue = {specs[-1].max():+.8f}")
dev = max(np.abs(s - specs[0]).max() for s in specs)
ok = dev < 1e-10
print(f"  max deviation across beta = {dev:.2e}  {'ok' if ok else 'MISMATCH'}")
results["TEST 4"] = ok

banner("TEST 5  ratio mu_3 / lambda_2 -> 3/2")
for kappa in [0.3, 0.1, 0.05, 0.02]:
    r = mu_closed(3, kappa).real / lambda_oop(2, kappa).real
    print(f"  kappa={kappa:5}: mu_3/lambda_2 = {r:.6f}")
ok = abs(mu_closed(3, 0.02).real / lambda_oop(2, 0.02).real - 1.5) < 0.015
print(f"  limit vs 1.5: {'ok' if ok else 'MISMATCH'}")
results["TEST 5"] = ok

banner("TEST 6  N-independence (incl. N not divisible by 4)")
ok = True
for N in [24, 30, 37, 41, 60]:
    J = inplane_jacobian(N, 1.0, 0.3)
    direct = jacobian_mode_eigenvalue(J, 3).real
    closed = mu_closed(3, 0.3).real
    good = abs(direct - closed) < 1e-5
    ok &= good
    print(f"  N={N:3d}: brute={direct:+.6f} closed={closed:+.6f} "
          f"{'ok' if good else 'MISMATCH'}")
results["TEST 6"] = ok

banner("TEST 7  nonlinear confirmation (full flow, no linearisation)")
def nonlinear_m3_rate(N=60, beta=1.0, eps=0.3, T=300, dt=0.01, amp=1e-7):
    phi = 2 * np.pi * np.arange(N) / N
    D = np.diag([0, 0, beta]) + eps * A_from_axis((0, 0, 1))
    ph = phi + amp * np.sin(3 * phi)
    X = np.stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)], 1)
    ts, qs = [], []
    for s in range(int(T / dt) + 1):
        if s % int(5 / dt) == 0:
            ang = np.arctan2(X[:, 1], X[:, 0])
            qs.append(np.abs(np.mean(np.exp(3j * ang)))); ts.append(s * dt)
        X = X + dt * flow(X, D); X /= np.linalg.norm(X, axis=1, keepdims=True)
    t = np.array(ts); q = np.array(qs); k = (t > T * 0.3) & (q > 1e-12)
    return np.polyfit(t[k], np.log(q[k]), 1)[0]
meas = nonlinear_m3_rate()
pred = mu_closed(3, 0.3).real
print(f"  measured nonlinear growth rate = {meas:.6f}")
print(f"  closed-form mu_3               = {pred:.6f}")
print(f"  ratio = {meas/pred:.4f}  (expect ~1; ~1% low is Euler dt bias)")
ok = abs(meas / pred - 1) < 0.05
results["TEST 7"] = ok

banner("SUMMARY")
for k, v in results.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
print("\nALL PASS" if all(results.values()) else "\nSOME FAILED")