"""Measured growth rates of the rotating ring vs the paper's closed forms
(Cor. "no-threshold", Prop. "oop-spectrum", Prop. "inplane-spectrum", Cor. "m3-fastest").

Model: true exponential flow, D = diag(0,0,beta) + eps*A, beta = 1, V = -Id,
projected Euler.  A = paper's A_sparse (omega = +e3), so kappa = eps (omega . e3) = eps.
Start: uniform equatorial ring phi_i = 2 pi i / N, plus a tiny seed (a = 1e-4):
  * out-of-plane m=2:  z_i = a cos(2 phi_i), renormalized onto the sphere;
    tracked c2 = |(2/N) sum z_i exp(-2 i phi_i)| (phi_i = current azimuth, so the
    rotation drops out).  Predicted lambda_2 = I_2(kappa)/I_0(kappa) (~ kappa^2/8).
  * in-plane m = 2, 3, 4:  phi_i -> phi_i + a cos(m phi_i), z = 0 (ring stays flat);
    tracked h_m = |(1/N) sum exp(i m phi_i)|.  Predicted Re mu_m from eq. inplane-eigen.
Fit: least squares of log(amplitude) vs t while the amplitude is between 3x and 100x its
starting value (the linear range).  If a mode never reaches 3x within the horizon, the
slope over the whole run is reported instead and marked '*'.
Extra checks: one case at dt = 0.025 (step independence), and the skewed matrix at
eps = 0.1 (kappa = 0.3) vs the sparse matrix at eps = 0.3.
Output: a table (stdout) and results/figures/growthRates.png.  Deterministic.
Runtime: ~10 min on 8 cores.
Run from inside export/:  PYTHONPATH=. python verification/growth_rate_check.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from multiprocessing import Pool
from scipy.special import iv

BETA, AMP = 1.0, 1e-4
A_SPARSE = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])    # omega = +e3
A_SKEWED = np.array([[0., -3, 0], [3, 0, -3], [0, 3, 0]])   # omega = (3,0,3)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "figures", "growthRates.png"))


# ---------------------------------------------------------------- predictions
def lam2(k):
    return iv(2, k) / iv(0, k)


def mu(m, k):
    """Eq. inplane-eigen with c_n = I_n(kappa) (-i)^n; returns the complex eigenvalue."""
    c = lambda n: iv(n, k) * (-1j) ** n
    g = lambda q: c(q)
    gp = lambda q: k / 2 * (c(q - 1) + c(q + 1))
    G = lambda q: (c(q - 1) - c(q + 1)) / (2j)
    Gp = lambda q: 0.5 * (c(q - 1) + c(q + 1)) + k / (4j) * (c(q - 2) - c(q + 2))
    Om = G(0) / g(0)
    return (Gp(0) - Gp(m) - Om * (gp(0) - gp(m))) / g(0)


# ---------------------------------------------------------------- simulation
def field(X, D):
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)
    return F - np.sum(X * F, axis=1)[:, None] * X


def run(job):
    """job = (label, kind, m, eps, which_A, N, dt, T). Returns (job, rate, n_fit_points, used_window)."""
    label, kind, m, eps, which, N, dt, T = job
    A = A_SPARSE if which == "sparse" else A_SKEWED
    D = np.diag([0, 0, BETA]) + eps * A
    phi = 2 * np.pi * np.arange(N) / N
    if kind == "oop":
        X = np.stack([np.cos(phi), np.sin(phi), AMP * np.cos(2 * phi)], 1)
    else:
        ph = phi + AMP * np.cos(m * phi)
        X = np.stack([np.cos(ph), np.sin(ph), np.zeros(N)], 1)
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    def amp(X):
        ph = np.arctan2(X[:, 1], X[:, 0])
        if kind == "oop":
            return abs(2 / N * np.sum(X[:, 2] * np.exp(-2j * ph)))
        return abs(np.mean(np.exp(1j * m * ph)))

    steps = int(round(T / dt))
    every = max(1, steps // 2000)
    ts, ys = [], []
    a0 = amp(X)
    for s in range(steps + 1):
        if s % every == 0:
            a = amp(X)
            ts.append(s * dt); ys.append(a)
            if a > 100 * a0:
                break
        X = X + dt * field(X, D)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    ts, ys = np.array(ts), np.array(ys)
    win = (ys >= 3 * a0) & (ys <= 100 * a0)
    used_window = win.sum() >= 5
    if not used_window:
        win = np.ones_like(ys, dtype=bool)
    rate = np.polyfit(ts[win], np.log(ys[win]), 1)[0]
    return job, rate, int(win.sum()), used_window


def horizon(eps, which):
    k = eps * (3.0 if which == "skewed" else 1.0)
    return 1.5 * np.log(100) / lam2(k)          # lambda_2 is the slowest growing mode here


def main():
    eps_list, N_list = [0.1, 0.2, 0.3, 0.5], [200, 1000]
    jobs = []
    for eps in eps_list:
        for N in N_list:
            T = horizon(eps, "sparse")
            jobs.append((f"sparse e={eps} N={N}", "oop", 2, eps, "sparse", N, 0.05, T))
            for m in (2, 3, 4):
                jobs.append((f"sparse e={eps} N={N}", "inp", m, eps, "sparse", N, 0.05, T))
    # step-size check and skewed-matrix check
    for kind, m in (("oop", 2), ("inp", 3)):
        jobs.append(("dt check e=0.3 N=200", kind, m, 0.3, "sparse", 200, 0.025, horizon(0.3, "sparse")))
        for N in N_list:
            jobs.append((f"skewed e=0.1 N={N}", kind, m, 0.1, "skewed", N, 0.05, horizon(0.1, "skewed")))
    jobs.sort(key=lambda j: -j[5] * j[7] / j[6])   # longest first
    with Pool(8) as pool:
        out = pool.map(run, jobs, chunksize=1)
    res = {(j[0], j[1], j[2]): (r, n, ok) for j, r, n, ok in out}

    def get(label, kind, m):
        r, n, ok = res[(label, kind, m)]
        return r, ("" if ok else "*")

    print(f"Model: D = diag(0,0,{BETA}) + eps*A, V = -Id, dt = 0.05, seed amplitude {AMP}")
    print("Predicted: lambda_2 = I_2/I_0 (Bessel) and kappa^2/8; mu_m = Re of eq. inplane-eigen")
    k_chk = 0.1
    print(f"Formula sanity at kappa = {k_chk}: mu_1 = {mu(1, k_chk).real:+.5f} (expect ~ -0.5), "
          f"mu_3 = {mu(3, k_chk).real:.6f} vs 3k^2/16 = {3 * k_chk ** 2 / 16:.6f}")
    print("Note: the formula gives Re mu_2 = Re mu_4 = 0 exactly (mu_2, mu_4 purely imaginary).\n")

    hdr = (f"{'eps':>4} {'kappa':>5} {'N':>5} | {'lam2 Bessel':>11} {'k^2/8':>9} {'lam2 meas':>10} {'ratio':>6} | "
           f"{'mu2 pred':>9} {'mu2 meas':>10} | {'mu3 pred':>9} {'mu3 meas':>10} {'ratio':>6} | "
           f"{'mu4 pred':>9} {'mu4 meas':>10}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for eps in eps_list:
        k = eps
        for N in N_list:
            lab = f"sparse e={eps} N={N}"
            l_m, f1 = get(lab, "oop", 2)
            m2, f2 = get(lab, "inp", 2); m3, f3 = get(lab, "inp", 3); m4, f4 = get(lab, "inp", 4)
            rows.append((k, N, l_m, m3))
            print(f"{eps:>4} {k:>5.2f} {N:>5} | {lam2(k):>11.3e} {k * k / 8:>9.3e} {l_m:>9.3e}{f1:1} "
                  f"{l_m / lam2(k):>6.3f} | {mu(2, k).real:>9.1e} {m2:>+9.1e}{f2:1} | {mu(3, k).real:>9.3e} "
                  f"{m3:>9.3e}{f3:1} {m3 / mu(3, k).real:>6.3f} | {mu(4, k).real:>9.1e} {m4:>+9.1e}{f4:1}")
    print("  * = never reached 3x its start within the horizon; slope over the whole run instead.")
    print("  ratio = measured / Bessel-predicted.\n")
    print("Reading mu_2 and mu_4: their eigenvalues are purely imaginary (oscillation, no growth).")
    print("Forward Euler inflates an oscillation of frequency w by |1 + i w dt|, i.e. a spurious")
    print("rate of w^2 dt/2. Compare the measured slopes with that artefact:")
    for eps in eps_list:
        r2, _ = get(f"sparse e={eps} N=1000", "inp", 2); r4, _ = get(f"sparse e={eps} N=1000", "inp", 4)
        print(f"  eps={eps}:  mu_2 meas {r2:.2e} vs (Im mu_2)^2 dt/2 = {mu(2, eps).imag ** 2 * 0.05 / 2:.2e};"
              f"  mu_4 meas {r4:.2e} vs (Im mu_4)^2 dt/2 = {mu(4, eps).imag ** 2 * 0.05 / 2:.2e}")
    print()

    print("Step-size check (eps = 0.3, N = 200):")
    a, _ = get("sparse e=0.3 N=200", "oop", 2); b, _ = get("dt check e=0.3 N=200", "oop", 2)
    c, _ = get("sparse e=0.3 N=200", "inp", 3); d, _ = get("dt check e=0.3 N=200", "inp", 3)
    print(f"  lambda_2: dt=0.05 {a:.4e}   dt=0.025 {b:.4e}   relative change {abs(a - b) / abs(b):.2%}")
    print(f"  mu_3    : dt=0.05 {c:.4e}   dt=0.025 {d:.4e}   relative change {abs(c - d) / abs(d):.2%}")
    print(f"  lambda_2 error halves with dt (first-order Euler error); extrapolating to dt -> 0,")
    print(f"  2*lambda(0.025) - lambda(0.05) = {2 * b - a:.4e}  vs  predicted I_2/I_0 = {lam2(0.3):.4e}"
          f"  ({abs(2 * b - a - lam2(0.3)) / lam2(0.3):.2%} off)\n")

    print("Same kappa, different matrix (kappa = 0.3): skewed eps=0.1 vs sparse eps=0.3")
    sk = []
    for N in N_list:
        ls, _ = get(f"skewed e=0.1 N={N}", "oop", 2); ms, _ = get(f"skewed e=0.1 N={N}", "inp", 3)
        lp, _ = get(f"sparse e=0.3 N={N}", "oop", 2); mp, _ = get(f"sparse e=0.3 N={N}", "inp", 3)
        sk.append((N, ls, ms))
        print(f"  N={N:>5}: lambda_2 skewed {ls:.4e} vs sparse {lp:.4e} (pred {lam2(0.3):.4e});"
              f"  mu_3 skewed {ms:.4e} vs sparse {mp:.4e} (pred {mu(3, 0.3).real:.4e})")

    # ------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kk = np.geomspace(0.07, 0.7, 200)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(kk, lam2(kk), color="#1f77b4", lw=1.4, label=r"$\lambda_2=I_2/I_0$ (out-of-plane)")
    ax.loglog(kk, [mu(3, x).real for x in kk], color="#d62728", lw=1.4, label=r"$\mu_3$ (in-plane)")
    for k, N, l_m, m3 in rows:
        mk = "o" if N == 1000 else "s"
        ax.loglog(k, l_m, mk, color="#1f77b4", ms=4, mfc="none" if N == 200 else "#1f77b4")
        ax.loglog(k, m3, mk, color="#d62728", ms=4, mfc="none" if N == 200 else "#d62728")
    for N, ls, ms in sk:
        ax.loglog(0.3, ls, "x", color="k", ms=5)
        ax.loglog(0.3, ms, "x", color="k", ms=5)
    ax.plot([], [], "o", color="0.4", ms=4, label="measured, N=1000")
    ax.plot([], [], "s", color="0.4", mfc="none", ms=4, label="measured, N=200")
    ax.plot([], [], "x", color="k", ms=5, label=r"skewed $A$, $\kappa=0.3$")
    ax.set_xlabel(r"coupling $\kappa=\varepsilon\,(\omega\cdot e_3)$")
    ax.set_ylabel("growth rate")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_xticks([0.1, 0.2, 0.3, 0.5])
    ax.set_xticks([], minor=True)
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5"])
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print("\nsaved results/figures/growthRates.png")


if __name__ == "__main__":
    main()
