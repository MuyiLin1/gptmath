"""E6: numerical checks of the corrected appendix statements and of Proposition "VD".

  1. Ripple law (Lemma "Exact height dissipation"): Ldot = -2 beta L^2 - G + O(L^3),
     G = |<z^2 e^{i phi}>|^2, and the +3 beta L^3 correction for z ~ cos 2phi.
  2. General near-ring law: Ldot = -2 zbar^2 + 2 zbar <z^3> - G - 2 beta L (L - zbar^2) + h.o.t.
  3. Polar lemma: eigenvalues at a single pole and at antipodal splits (rate c_+- inside a cluster).
  4. Proposition "VD": with V = -D the field has zero circulation around loops for any D,
     and the uniform equatorial ring is exactly stationary.
Deterministic, < 1 min.  Run from inside export/:  PYTHONPATH=. python verification/lemma_fix_check.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "review"))
os.makedirs(OUT, exist_ok=True)
A_SPARSE = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
res = {}


def ldot_exact(z, phi, beta):
    r = np.sqrt(1 - z ** 2)
    X = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    a = np.exp(beta * np.outer(z, z))
    a /= a.sum(1, keepdims=True)
    return -2 / len(z) * np.sum(a * (np.outer(z, z) - (z ** 2)[:, None] * (X @ X.T)))


print("1. Zero-sum ripple: measured Ldot vs -2 beta L^2 - G", flush=True)
N = 400
phi = 2 * np.pi * np.arange(N) / N
rows = []
for name, prof in [("cos2", np.cos(2 * phi)), ("cos3", np.cos(3 * phi)), ("cos2+cos3", np.cos(2 * phi) + np.cos(3 * phi))]:
    for beta in [0.0, 0.5, 1.0, 2.0]:
        for amp in [0.02, 0.01]:
            z = amp * prof
            L = np.mean(z ** 2)
            G = abs(np.mean(z ** 2 * np.exp(1j * phi))) ** 2
            meas, pred = ldot_exact(z, phi, beta), -2 * beta * L ** 2 - G
            cubic = (meas + 2 * beta * L ** 2 + G) / L ** 3
            rows.append(dict(profile=name, beta=beta, amp=amp, Ldot=meas, pred=pred, rel_err_vs_L2=(meas - pred) / L ** 2,
                             cubic_coeff=cubic))
            print(f"  {name:10s} beta={beta:3.1f} amp={amp:5.3f}  Ldot={meas:+.4e} pred={pred:+.4e}  "
                  f"(Ldot-pred)/L^2={(meas - pred) / L ** 2:+.2e}  cubic coeff={cubic:+.3f}", flush=True)
res["ripple"] = rows
print("  expected: (Ldot-pred)/L^2 -> 0 as amp -> 0; cubic coeff = 3*beta for cos2\n", flush=True)

print("2. General near-ring law (no tilt): z = delta + a cos2phi + b cos3phi", flush=True)
rows = []
for beta in [0.5, 1.0]:
    for delta, a, b in [(0.01, 0.01, 0.0), (0.005, 0.01, 0.01), (0.002, 0.02, 0.01)]:
        z = delta + a * np.cos(2 * phi) + b * np.cos(3 * phi)
        L, zb, z3 = np.mean(z ** 2), np.mean(z), np.mean(z ** 3)
        G = abs(np.mean(z ** 2 * np.exp(1j * phi))) ** 2
        pred = -2 * zb ** 2 + 2 * zb * z3 - G - 2 * beta * L * (L - zb ** 2)
        meas = ldot_exact(z, phi, beta)
        rows.append(dict(beta=beta, delta=delta, a=a, b=b, Ldot=meas, pred=pred, rel_err_vs_L2=(meas - pred) / L ** 2))
        print(f"  beta={beta} delta={delta} a={a} b={b}: Ldot={meas:+.4e} pred={pred:+.4e} "
              f"(Ldot-pred)/L^2={(meas - pred) / L ** 2:+.2e}  sign ok: {meas <= 0}", flush=True)
res["general"] = rows
print()


def field(X, D, V):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = W @ (X @ V.T)
    return F - np.sum(X * F, axis=1)[:, None] * X


def tangent_jacobian(X, D, V, h=1e-6):
    n = len(X)
    B = []
    for x in X:
        u = np.cross(x, [0.3, 0.5, 0.8]); u /= np.linalg.norm(u)
        B.append(np.stack([u, np.cross(x, u)], 1))
    J = np.zeros((2 * n, 2 * n))
    for j in range(n):
        for k in range(2):
            Xp, Xm = X.copy(), X.copy()
            Xp[j] += h * B[j][:, k]; Xp[j] /= np.linalg.norm(Xp[j])
            Xm[j] -= h * B[j][:, k]; Xm[j] /= np.linalg.norm(Xm[j])
            dF = (field(Xp, D, V) - field(Xm, D, V)) / (2 * h)
            for i in range(n):
                J[2 * i:2 * i + 2, 2 * j + k] = B[i].T @ dF[i]
    return J


print("3. Polar configurations (beta = 2, V = -Id, eps = 0)", flush=True)
beta = 2.0
D = np.diag([0, 0, beta])
rows = []
for nN, nS in [(10, 0), (5, 5), (3, 7), (2, 8), (1, 9)]:
    X = np.array([[0, 0, 1.0]] * nN + [[0, 0, -1.0]] * nS)
    ev = np.linalg.eigvals(tangent_jacobian(X, D, -np.eye(3))).real
    c = lambda a, b: (a * np.exp(beta) - b * np.exp(-beta)) / (a * np.exp(beta) + b * np.exp(-beta))
    pred = {"c_north": c(nN, nS) if nN >= 2 else None, "c_south": c(nS, nN) if nS >= 2 else None}
    distinct = sorted(set(np.round(ev, 4).tolist()))
    rows.append(dict(north=nN, south=nS, eig_min=float(ev.min()), eig_max=float(ev.max()), n_pos=int((ev > 1e-6).sum()),
                     n_neg=int((ev < -1e-6).sum()), distinct=distinct, **pred))
    print(f"  {nN}/{nS}: #pos={rows[-1]['n_pos']:2d} #neg={rows[-1]['n_neg']:2d} distinct={distinct}  predicted c+-={pred}",
          flush=True)
res["poles"] = rows
print("  expected: single pole {0, +1}; every split has a positive eigenvalue; predicted c+- appear in the list\n",
      flush=True)

print("4. Proposition VD: circulation of V=-D field vs V=-Id field around great circles", flush=True)
rng = np.random.default_rng(0)
rows = []
t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
for trial in range(5):
    Dr = rng.normal(size=(3, 3))
    Y = rng.normal(size=(300, 3)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    loop = np.cos(t)[:, None] * Q[:, 0] + np.sin(t)[:, None] * Q[:, 1]
    tang = -np.sin(t)[:, None] * Q[:, 0] + np.cos(t)[:, None] * Q[:, 1]
    circ = {}
    for name, V in [("-D", -Dr), ("-Id", -np.eye(3))]:
        S = loop @ Dr @ Y.T
        S -= S.max(1, keepdims=True)
        W = np.exp(S); W /= W.sum(1, keepdims=True)
        F = W @ (Y @ V.T)
        F -= np.sum(loop * F, 1)[:, None] * loop
        circ[name] = float(np.sum(F * tang) * 2 * np.pi / len(t))
    rows.append(circ)
    print(f"  random D #{trial}: circulation V=-D {circ['-D']:+.2e}   V=-Id {circ['-Id']:+.3f}", flush=True)
ring = np.stack([np.cos(t[::10]), np.sin(t[::10]), 0 * t[::10]], 1)
ring_speed = {}
for eps in [0.1, 0.3, 1.0, 3.0]:
    Dm = np.diag([0, 0, 1.0]) + eps * A_SPARSE
    ring_speed[eps] = float(np.abs(field(ring, Dm, -Dm)).max())
    print(f"  uniform equatorial ring, eps={eps}: max |velocity| under V=-D = {ring_speed[eps]:.1e}", flush=True)
res["VD"] = dict(circulations=rows, ring_max_speed=ring_speed)

with open(os.path.join(OUT, "E6_lemma_checks.json"), "w") as f:
    json.dump(res, f, indent=1)
print("\nsaved results/review/E6_lemma_checks.json", flush=True)
