"""E9: rotating triangles with unequal clusters (Remark "Unequal triangles", Appendix "Proof of kstab").

Setting: D = diag(0,0,beta) + eps*A_sparse, V = -Id, beta = 1, N = 60 tokens split n0/n1/n2 on the equator.
(1) Find the relative equilibrium: cluster azimuths 0, a1, a2 such that all three clusters turn at the
    same rate (Newton from the equilateral triangle).
(2) Jacobian of the flow in the co-rotating frame by central finite differences in (azimuth, latitude)
    of all N tokens; report the rotation rate, the eigenvalues on the imaginary axis (phase 0, tilt
    +-i Omega), and the largest real part among the others.
Run from inside export/:  PYTHONPATH=. python verification/unequal_triangle_check.py
"""
import json
import os
import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "review"))
A = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
BETA = 1.0


def velocity(ph, th, eps):
    X = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)], 1)
    D = np.diag([0., 0., BETA]) + eps * A
    S = X @ D @ X.T
    S -= S.max(1, keepdims=True)
    W = np.exp(S); W /= W.sum(1, keepdims=True)
    F = -(W @ X)
    F -= (F * X).sum(1, keepdims=True) * X
    e_ph = np.stack([-np.sin(ph), np.cos(ph), 0 * ph], 1)
    e_th = np.stack([-np.sin(th) * np.cos(ph), -np.sin(th) * np.sin(ph), np.cos(th)], 1)
    return (F * e_ph).sum(1) / np.cos(th), (F * e_th).sum(1)


def equilibrium(sizes, eps):
    lab = np.repeat(np.arange(3), sizes)

    def rates(a):
        ph = np.array([0.0, a[0], a[1]])[lab]
        w = velocity(ph, np.zeros(len(lab)), eps)[0]
        return np.array([w[lab == c].mean() for c in range(3)])

    a = fsolve(lambda a: rates(a)[1:] - rates(a)[0], [2 * np.pi / 3, 4 * np.pi / 3], xtol=1e-14)
    r = rates(a)
    return np.array([0.0, a[0], a[1]])[lab], float(r.mean()), float(np.ptp(r)), a


def spectrum(ph, Om, eps, h=1e-6):
    N = len(ph)

    def G(v):
        a, b = velocity(v[:N], v[N:], eps)
        return np.concatenate([a - Om, b])
    v0 = np.concatenate([ph, np.zeros(N)])
    J = np.zeros((2 * N, 2 * N))
    for i in range(2 * N):
        e = np.zeros(2 * N); e[i] = h
        J[:, i] = (G(v0 + e) - G(v0 - e)) / (2 * h)
    return np.linalg.eigvals(J)


def main():
    os.makedirs(OUT, exist_ok=True)
    splits = [(20, 20, 20), (21, 20, 19), (21, 21, 18), (22, 20, 18), (24, 20, 16), (26, 20, 14),
              (30, 20, 10), (36, 16, 8), (40, 15, 5), (50, 6, 4)]
    out = []
    print(f"{'split':>10} {'eps':>4} | {'angles (deg)':>15} {'Omega':>8} {'resid':>8} | "
          f"{'phase':>8} {'tilt':>17} | {'max Re other':>12} {'#Re>1e-7':>8}")
    for eps in (0.1, 0.3, 0.6, 1.0):
        for sz in splits:
            ph, Om, resid, a = equilibrium(sz, eps)
            ev = spectrum(ph, Om, eps)
            i0 = int(np.argmin(np.abs(ev)))
            rest = np.delete(ev, i0)
            it = np.argsort(np.abs(rest - 1j * Om))[0], np.argsort(np.abs(rest + 1j * Om))[0]
            tilt = rest[list(it)]
            other = np.delete(rest, list(it))
            r = dict(split=sz, eps=eps, angles=np.degrees(a).tolist(), Omega=Om, resid=resid,
                     phase=abs(ev[i0]), tilt=[complex(t).imag for t in tilt],
                     tilt_re=float(np.abs(tilt.real).max()), max_re_other=float(other.real.max()),
                     n_pos=int((ev.real > 1e-7).sum()))
            out.append(r)
            print(f"{'/'.join(map(str, sz)):>10} {eps:4.1f} | {a[0] * 180 / np.pi:7.2f} {a[1] * 180 / np.pi:7.2f} "
                  f"{Om:+8.5f} {resid:8.1e} | {r['phase']:8.1e} {tilt[0].real:+.0e}{tilt[0].imag:+.4f}i | "
                  f"{r['max_re_other']:+12.3e} {r['n_pos']:8d}", flush=True)
        print()
    with open(os.path.join(OUT, "E9_unequal_triangle.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results/review/E9_unequal_triangle.json")


if __name__ == "__main__":
    main()
