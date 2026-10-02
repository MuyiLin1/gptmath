"""Isotropic case: a true gradient flow, broken by antisymmetry.

With an isotropic symmetric part D_s = beta*I and value matrix V = -Id, the
eps = 0 flow is EXACTLY the gradient flow of the interaction energy
    E(mu) = 1/2 * mean_{i,j} exp(X_i . D X_j),
with mobility 1/(beta * Z_i) (Z_i = softmax normaliser).  This is the case in
which "antisymmetry breaks the gradient-flow structure" is literally true, and
it is the setting of the paper's non-gradient theorem (whose witness D_s = 0 is
the beta -> 0 limit).

The script runs the true (full-exponential) flow from the same random start for
eps in {0, 0.05, 0.1} with the sparse A (dual axis e3) and records:
  (a) the energy E_eps(t)            -> decreases and settles; the settled values
                                        differ only at order eps^2 ("the energy is blind
                                        at first order", Theorem "energy is even")
  (b) the mean speed of the tokens    -> goes to 0 at eps = 0, stays > 0 for eps > 0
  (c) cumulative rotation about e3    -> 0 at eps = 0, grows linearly (rate ~ eps)

Output: results/figures/isotropic.png and a printed summary.
Run from inside export/:  PYTHONPATH=. python figures/plot_isotropic.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
N, BETA, DT, STEPS = 300, 1.0, 0.05, 4000
EPS_LIST = [0.0, 0.05, 0.10]
A_SPARSE = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])   # dual axis omega = e3
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "figures", "isotropic.png"))


def velocity(X, D):
    """True self-attention field with V = -Id, projected onto the sphere."""
    S = X @ D @ X.T
    S -= S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)
    return F - np.sum(X * F, axis=1)[:, None] * X


def energy(X, D):
    return 0.5 * np.mean(np.exp(X @ D @ X.T))


def run(eps):
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    D = BETA * np.eye(3) + eps * A_SPARSE
    E, speed, angle = [], [], [0.0]
    for _ in range(STEPS):
        V = velocity(X, D)
        E.append(energy(X, D))
        speed.append(np.linalg.norm(V, axis=1).mean())
        rho2 = X[:, 0] ** 2 + X[:, 1] ** 2
        spin = np.sum(X[:, 0] * V[:, 1] - X[:, 1] * V[:, 0]) / np.sum(rho2)
        angle.append(angle[-1] + DT * spin)
        X = X + DT * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.array(E), np.array(speed), np.array(angle[1:])


def main():
    t = DT * np.arange(STEPS)
    res = {eps: run(eps) for eps in EPS_LIST}
    print(f"isotropic D_s = {BETA}*I, V = -Id, N = {N}, dt = {DT}, steps = {STEPS}, seed = {SEED}")
    print(f"{'eps':>6} {'E start':>10} {'E end':>10} {'max E rise':>12} {'final speed':>12} {'spin rate':>10}")
    for eps, (E, sp, ang) in res.items():
        rate = (ang[-1] - ang[STEPS // 2]) / (t[-1] - t[STEPS // 2])
        print(f"{eps:>6.2f} {E[0]:>10.5f} {E[-1]:>10.5f} {np.max(np.diff(E)):>12.1e} "
              f"{sp[-1]:>12.2e} {rate:>10.5f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for eps, (E, sp, ang) in res.items():
        lab = rf"$\varepsilon={eps:.2f}$"
        ax[0].plot(t, E, label=lab)
        ax[1].semilogy(t, sp, label=lab)
        ax[2].plot(t, ang, label=lab)
    ax[0].set(title="(a) energy settles; values differ only at $O(\\varepsilon^2)$", xlabel="time $t$",
              ylabel=r"$\mathcal{E}_\varepsilon$")
    ax[1].set(title="(b) tokens stop only at $\\varepsilon=0$", xlabel="time $t$",
              ylabel="mean token speed")
    ax[2].set(title="(c) persistent rotation, rate $\\propto\\varepsilon$", xlabel="time $t$",
              ylabel="cumulative rotation about $e_3$")
    for a in ax:
        a.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("saved results/figures/isotropic.png")


if __name__ == "__main__":
    main()
