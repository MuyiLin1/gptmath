"""Linear stability of the rotating k-cluster states (Proposition "kstab").

Setting (paper conventions): D = diag(0,0,beta) + eps*A, A = A_sparse = [[0,-1,0],[1,0,0],[0,0,0]]
(dual axis omega = +e3, so kappa = eps), V = -Id, true exponential softmax.
For each k, N tokens are split into k equal clusters at the vertices of a regular k-gon on the
equator. We (1) check the state is a relative equilibrium rotating at Omega_k, (2) compute the
Jacobian of the flow in the co-rotating frame by central finite differences in (azimuth, latitude)
coordinates, and (3) compare its 2N eigenvalues with the closed-form spectrum of the paper's
proof (Appendix "Proof of Proposition kstab"). Finally (4) a tilted, noisy triangle is simulated.
Runs in about a minute on a laptop CPU. Requires numpy and scipy.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

A = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
BETA = 1.0

def positions(ph, th):
    return np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)], 1)

def velocity(ph, th, eps, beta=BETA):
    """(azimuth rate, latitude rate) of every token under the true flow."""
    X = positions(ph, th)
    D = np.diag([0., 0., beta]) + eps * A
    S = X @ D @ X.T
    S -= S.max(1, keepdims=True)
    W = np.exp(S); W /= W.sum(1, keepdims=True)
    F = -(W @ X)                                   # V = -Id
    F -= (F * X).sum(1, keepdims=True) * X         # tangential projection
    e_ph = np.stack([-np.sin(ph), np.cos(ph), 0 * ph], 1)
    e_th = np.stack([-np.sin(th) * np.cos(ph), -np.sin(th) * np.sin(ph), np.cos(th)], 1)
    return (F * e_ph).sum(1) / np.cos(th), (F * e_th).sum(1)

def cluster_state(k, N):
    return np.repeat(2 * np.pi * np.arange(k) / k, N // k), np.zeros(N)

def numeric_spectrum(k, N, eps, h=1e-6):
    ph, th = cluster_state(k, N)
    dph, dth = velocity(ph, th, eps)
    Om = dph.mean()
    resid = max(np.abs(dph - Om).max(), np.abs(dth).max())
    def G(v):
        a, b = velocity(v[:N], v[N:], eps)
        return np.concatenate([a - Om, b])
    v0 = np.concatenate([ph, th]); J = np.zeros((2 * N, 2 * N))
    for i in range(2 * N):
        e = np.zeros(2 * N); e[i] = h
        J[:, i] = (G(v0 + e) - G(v0 - e)) / (2 * h)
    return Om, resid, np.linalg.eigvals(J)

def formula_spectrum(k, N, kap):
    th = 2 * np.pi * np.arange(k) / k; psi = -th
    g = np.exp(kap * np.sin(psi)); Z = g.sum(); p = g / Z
    Om = -(p * np.sin(th)).sum()
    H = np.cos(psi) * g * (1 + kap * np.sin(psi) - Om * kap)      # G' - Omega g'
    lam_in = H.sum() / Z
    cbar = (p * np.cos(th)).sum()
    nu = [(H * (1 - np.exp(1j * q * th))).sum() / Z for q in range(k)]
    mu = [cbar - (p * np.exp(1j * q * th)).sum() for q in range(k)]
    ev = nu + mu + [lam_in] * (N - k) + [cbar] * (N - k)
    return Om, np.array(ev, complex), lam_in, cbar, nu, mu

def match_error(a, b):
    C = np.abs(a[:, None] - b[None, :]); r, c = linear_sum_assignment(C)
    return C[r, c].max()

print("Part 1. Numerical Jacobian vs closed-form spectrum")
print(f"{'N':>4} {'k':>2} {'eps':>4} | {'Omega_k':>8} {'RE resid':>8} | {'#Re>0':>5} {'max Re':>10} | {'max |num-formula|':>17}")
for N in (60, 120):
    for eps in (0.1, 0.3, 0.6, 1.0):
        for k in (2, 3, 4, 5, 6):
            Om, resid, ev = numeric_spectrum(k, N, eps)
            Omf, evf, *_ = formula_spectrum(k, N, eps)
            npos = int((ev.real > 1e-7).sum())
            print(f"{N:4d} {k:2d} {eps:4.1f} | {Om:+8.4f} {resid:8.1e} | {npos:5d} {ev.real.max():+10.3e} | {match_error(ev, evf):17.1e}")
    print()

print("Part 2. Closed-form rates for the triangle (k = 3), all must be < 0 except phase and tilt")
s = np.sqrt(3) / 2
for eps in (0.1, 0.3, 0.6, 1.0):
    Om, _, lam_in, cbar, nu, mu = formula_spectrum(3, 60, eps)
    C = np.cosh(eps * s)
    print(f"eps={eps:3.1f}: Omega_3={Om:.4f}  in-plane within-cluster={lam_in:+.4f}  "
          f"out-of-plane within-cluster={cbar:+.4f} (formula (1-C)/(1+2C)={(1 - C) / (1 + 2 * C):+.4f})  "
          f"centres: Re nu_1={nu[1].real:+.4f}, mu_0={mu[0].real:+.4f}, tilt mu_(+-1)={mu[1]:+.4f}")

print("\nPart 3. k >= 4: growing out-of-plane mode q = 2 vs eps^2/8 (eps^2/4 for k = 4)")
for k in (4, 5, 6, 7, 8):
    row = []
    for eps in (0.05, 0.1, 0.2):
        mu2 = formula_spectrum(k, 120, eps)[5][2].real
        pred = eps ** 2 / (4 if k == 4 else 8)
        row.append(f"eps={eps}: Re mu_2={mu2:.3e} pred={pred:.3e} ratio={mu2 / pred:.4f}")
    print(f"k={k}: " + " | ".join(row))

print("\nPart 4. A tilted, noisy triangle (N=60, eps=0.3, 5 deg tilt, noise 1e-3, dt=0.05)")
rng = np.random.default_rng(0); N, k, eps, dt = 60, 3, 0.3, 0.05
ph, _ = cluster_state(k, N); X = positions(ph, 0 * ph)
d = np.radians(5); R = np.array([[1, 0, 0], [0, np.cos(d), -np.sin(d)], [0, np.sin(d), np.cos(d)]])
X = X @ R.T + 1e-3 * rng.normal(size=(N, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
D = np.diag([0., 0., BETA]) + eps * A
for step in range(int(3000 / dt) + 1):
    if step % int(500 / dt) == 0:
        w, v = np.linalg.eigh(X.T @ X / N); tilt = np.degrees(np.arccos(abs(v[2, 0])))
        spread = np.mean([np.linalg.norm(X[c * 20:(c + 1) * 20] - X[c * 20:(c + 1) * 20].mean(0), axis=1).mean() for c in range(k)])
        print(f"  t={step * dt:6.0f}  plane tilt={tilt:.3f} deg  within-cluster spread={spread:.1e}")
    S = X @ D @ X.T; S -= S.max(1, keepdims=True); W = np.exp(S); W /= W.sum(1, keepdims=True)
    F = -(W @ X); F -= (F * X).sum(1, keepdims=True) * X
    X = X + dt * F; X /= np.linalg.norm(X, axis=1, keepdims=True)
