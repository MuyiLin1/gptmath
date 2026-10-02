"""Figure: the out-of-plane buckle is an instability of the flat ring, not a forced
O(eps^2) deformation.

We seed a tiny m=2 out-of-plane ripple on the flat ring and watch its coherent,
rotation-invariant amplitude c2(t) evolve under the true self-attention flow.

The m = 2 out-of-plane mode grows at every eps != 0, at rate
lambda_2 = I_2(kappa)/I_0(kappa) ~ kappa^2/8 (Cor. "no-threshold").  Over the finite
window T = 250 this looks like a threshold (apparent eps_c ~ 0.15 aligned, ~ 0.05
skewed); that threshold is an artefact of the window (Remark "finite-time artefact").

  (a) c2(t)/c2(0) for several eps (aligned A_sparse): slow growth at small eps,
      faster growth as eps rises.
  (b) growth factor c2(T)/c2(0) versus eps for the aligned and skewed matrices; the
      dashed lines mark where it first doubles by T = 250 (the apparent thresholds).
      They differ because the skewed matrix (Experiments section) has three times
      the projection omega . e3, so it reaches the same kappa = eps (omega . e3) at
      three times smaller eps.

Saves buckleThreshold.png to results/figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 1000
SEED_AMP = 5e-3
TAU = 0.05
STEPS = 5000
CHECKS = 100
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "figures", "buckleThreshold.png"))

# dual axis -e3: the transpose of the paper's A_sparse; only the sense of rotation differs
A_SPARSE = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 0]])
A_SKEWED = np.array([[0., -3, 0], [3, 0, -3], [0, 3, 0]])   # paper's A_skewed (Experiments section)
DS = np.diag([0., 0., 1.])


def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def Px(x, M):
    return M - (M * x).sum(1, keepdims=True) * x


def field(X, D):
    S = X @ D @ X.T
    S -= S.max(1, keepdims=True)
    K = np.exp(S)
    K /= K.sum(1, keepdims=True)
    return -Px(X, K @ X)


def c2_mode(X):
    C = np.cov(X.T)
    w, V = np.linalg.eigh(C)
    nu, e1, e2 = V[:, 0], V[:, 2], V[:, 1]
    h = X @ nu
    phi = np.arctan2(X @ e2, X @ e1)
    return 2 * abs(np.mean(h * np.exp(-2j * phi)))


def evolve_series(A, eps):
    phi = 2 * np.pi * np.arange(N) / N
    z0 = SEED_AMP * np.cos(2 * phi)
    r = np.sqrt(np.maximum(1 - z0 ** 2, 0))
    X = norm(np.c_[r * np.cos(phi), r * np.sin(phi), z0])
    D = DS + eps * A
    every = max(1, STEPS // CHECKS)
    ts, cs = [0.0], [c2_mode(X)]
    for k in range(1, STEPS + 1):
        X = norm(X + TAU * field(X, D))
        if k % every == 0:
            ts.append(k * TAU)
            cs.append(c2_mode(X))
    return np.array(ts), np.array(cs)


def doubling_eps(eps_grid, factors):
    lf, tgt = np.log(np.clip(factors, 1e-10, None)), np.log(2.0)
    for i in range(1, len(eps_grid)):
        if (lf[i - 1] - tgt) * (lf[i] - tgt) <= 0 and lf[i] != lf[i - 1]:
            return eps_grid[i - 1] + (tgt - lf[i - 1]) * (
                eps_grid[i] - eps_grid[i - 1]) / (lf[i] - lf[i - 1])
    return None


def main():
    aligned_grid = np.array([0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30])
    # skewed x3: apparent threshold ~0.05; stop at 0.12 (above that the ring has fragmented)
    skewed_grid  = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12])

    print("running aligned sweep ...")
    al = {e: evolve_series(A_SPARSE, float(e)) for e in aligned_grid}
    print("running skewed sweep ...")
    sk = {e: evolve_series(A_SKEWED,  float(e)) for e in skewed_grid}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # panel (a): aligned time traces
    trace_eps = [0.05, 0.10, 0.16, 0.20, 0.30]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(trace_eps)))
    for eps, c in zip(trace_eps, cmap):
        ts, cs = al[eps]
        axA.semilogy(ts, cs / cs[0], color=c, lw=1.9, label=rf"$\varepsilon={eps:.2f}$")
    axA.axhline(1.0, color="0.6", ls=":", lw=1.0)
    axA.set_xlabel("time $t$")
    axA.set_ylabel(r"$m{=}2$ amplitude  $c_2(t)/c_2(0)$")
    axA.set_title(r"(a) aligned matrix: grows at every $\varepsilon$, faster as $\varepsilon$ rises")
    axA.legend(frameon=False, fontsize=9, loc="upper left")

    # panel (b): growth factor vs eps
    thresholds = {}
    for grid, series, name, col, mk in [
        (aligned_grid, al, "aligned", "#1f77b4", "o"),
        (skewed_grid,  sk, "skewed",  "#d62728", "s"),
    ]:
        factors = np.array([series[e][1][-1] / series[e][1][0] for e in grid])
        axB.plot(grid, factors, mk + "-", color=col, ms=4.5, lw=1.7, label=name)
        ec = doubling_eps(grid, factors)
        thresholds[name] = ec
        if ec is not None:
            print(f"{name}: apparent (doubling by T=250) threshold eps_c ~ {ec:.3f}")
            axB.axvline(ec, color=col, ls="--", lw=1.3, alpha=0.75)

    axB.axhline(1.0, color="0.5", ls=":", lw=1.0)
    axB.axhline(2.0, color="0.5", ls="--", lw=0.9, label="doubling (×2)")
    axB.set_yscale("log")
    axB.set_xlabel(r"perturbation strength  $\varepsilon$")
    axB.set_ylabel(r"growth factor  $c_2(T)/c_2(0)$")
    axB.set_title(r"(b) apparent thresholds (doubling by $T=250$)")
    axB.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    ec_al = thresholds.get("aligned")
    ec_sk = thresholds.get("skewed")
    print("saved", OUT)
    if ec_al and ec_sk:
        print(f"apparent thresholds (T=250): aligned eps_c ~ {ec_al:.3f},  skewed eps_c ~ {ec_sk:.3f}")


if __name__ == "__main__":
    main()
