"""Figure: the height thermometer L = <(omega . x)^2> drains to zero via the two
mechanisms of the exact height-dissipation lemma (aligned case, Ds=diag(0,0,1)):

  (a) a rigid latitude shift of the ring relaxes EXPONENTIALLY, dot L = -2 L
      (L(t) = L0 e^{-2t}); a straight line on a log-y axis.
  (b) a zero-sum m=2 ring ripple relaxes ALGEBRAICALLY via the cubic law
      dot L = -2 beta L^2, i.e. 1/L is linear in t with slope 2 beta,
      and the law is independent of N.

Both curves show L monotonically decreasing to zero: tokens nudged off the ring
slide back. Saves heightDecay.png (both panels; paper Fig. "heightDecay") and
heightDecay_b.png (panel (b) alone; paper Fig. "ripple") to results/figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)

BETA = 1.0                                  # inverse temperature (eq. 1 convention)
DS = torch.diag(torch.tensor([0.0, 0.0, 1.0]))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "figures", "heightDecay.png"))
OUT_B = os.path.join(os.path.dirname(OUT), "heightDecay_b.png")


def proj_tang(x, y):
    return y - (y * x).sum(-1, keepdim=True) * x


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def V0(x):
    """Symmetric (eps=0) attention field, V = -Id, projected to the tangent."""
    K = torch.softmax(BETA * (x @ (DS @ x.T)), dim=1)
    return -proj_tang(x, K @ x)


def uniform_ring(n):
    phi = 2 * torch.pi * torch.arange(n) / n
    return phi, torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros(n)], 1)


def L_of(x):
    return x[:, 2].pow(2).mean().item()


def evolve(X0, tau, steps, checks):
    x = normalize(X0.clone())
    every = max(1, steps // checks)
    ts, Ls = [0.0], [L_of(x)]
    for k in range(1, steps + 1):
        x = normalize(x + tau * V0(x))
        if k % every == 0:
            ts.append(k * tau)
            Ls.append(L_of(x))
    return np.array(ts), np.array(Ls)


def linfit(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return slope, intercept, r2


def main():
    Ns = [48, 192, 768]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    styles = [("-", 3.4), ("--", 2.0), (":", 1.5)]   # distinct so overlap is visible
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Panel (a): rigid latitude shift -> exponential drain, rate 2 ----
    alpha = 0.12
    for N, c, (ls, lw) in zip(Ns, colors, styles):
        phi, X = uniform_ring(N)
        zhat = torch.tensor([0.0, 0.0, 1.0])
        X0 = X + alpha * zhat                       # uniform shift off the equator
        ts, Ls = evolve(X0, tau=0.02, steps=600, checks=200)
        axA.semilogy(ts, Ls, color=c, ls=ls, lw=lw, label=f"$N={N}$")
    # reference slope -2 line (L = L0 e^{-2t})
    tref = np.linspace(0, ts[-1], 50)
    L0 = alpha ** 2
    axA.semilogy(tref, L0 * np.exp(-2 * tref), "k-.", lw=1.0, label=r"$e^{-2t}$")
    axA.set_xlabel("time $t$")
    axA.set_ylabel(r"height $L=\langle(\omega\cdot x)^2\rangle$")
    axA.set_title("(a) rigid shift: exponential drain")
    axA.legend(frameon=False, fontsize=9)

    # ---- Panel (b): zero-sum m=2 ripple -> cubic law, 1/L linear ----
    eps = 0.10
    slope_last = r2_last = None
    for N, c, (ls, lw) in zip(Ns, colors, styles):
        phi, X = uniform_ring(N)
        zhat = torch.tensor([0.0, 0.0, 1.0])
        b = torch.cos(2 * phi)
        b = b - b.mean()                            # zero-sum (neutral) mode
        X0 = X + eps * b[:, None] * zhat
        ts, Ls = evolve(X0, tau=0.02, steps=30000, checks=250)
        axB.plot(ts, 1.0 / Ls, color=c, ls=ls, lw=lw, label=f"$N={N}$")
        slope_last, _, r2_last = linfit(ts, 1.0 / Ls)
    axB.set_xlabel("time $t$")
    axB.set_ylabel(r"$1/L$")
    axB.set_title("(b) zero-sum ripple: cubic law")
    axB.legend(frameon=False, fontsize=9, loc="upper left")
    axB.text(0.97, 0.05,
             rf"slope $=2\beta={slope_last:.2f}$" + "\n" + rf"$R^2={r2_last:.6f}$",
             transform=axB.transAxes, ha="right", va="bottom", fontsize=10,
             bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}")
    # panel (b) on its own, cut from the same figure (paper Fig. "ripple")
    bbox = axB.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(OUT_B, dpi=200, bbox_inches=bbox.expanded(1.02, 1.02))
    print("saved results/figures/heightDecay_b.png")
    print(f"panel (b) cubic-law fit: slope/beta = {slope_last / BETA:.3f}"
          f"  R^2 = {r2_last:.6f}")


if __name__ == "__main__":
    main()
