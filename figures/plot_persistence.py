"""Figure: the non-equilibrium steady state persists.

Starting from a random cloud, the true (un-truncated) self-attention flow settles
onto the equatorial ring. Two signatures (Theorem "rotating relative equilibrium")
separate the symmetric flow from the perturbed one:

  (a) Mobility-weighted kinetic energy  int int e^{x.Dy} ||V||^2 dmu dmu  versus time.
      At eps=0 it collapses to zero as the tokens come to rest on a static ring.
      At eps>0 it plateaus at a nonzero value -- the tokens keep circulating.
      (With D_s = diag(0,0,1) the eps=0 flow is not a gradient flow; it simply
      comes to rest. The isotropic gradient-flow case is figures/plot_isotropic.py.)

  (b) Cumulative mean rotation angle theta(t) about omega.
      At eps=0 it flattens (no residual motion); at eps>0 it grows linearly with
      slope ~ c*eps -- a steady, persistent spin.

Ds = diag(0,0,1), sparse A (dual axis omega=e3), V=-Id.  Saves persistence.png to
results/figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.float64)

BETA = 1.0
N = 400
DS = torch.diag(torch.tensor([0.0, 0.0, 1.0]))
A_SPARSE = torch.tensor([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])          # dual axis omega = e3 (paper's A_sparse)
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "results", "figures", "persistence.png"))


def proj_tang(x, y):
    return y - (y * x).sum(-1, keepdim=True) * x


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def field_and_kinetic_energy(x, eps):
    """True exponential flow.  Returns (V, mobility-weighted kinetic energy)."""
    D = DS + eps * A_SPARSE
    S = BETA * (x @ (D @ x.T))
    Kun = torch.exp(S - S.max(dim=1, keepdim=True).values)   # stable, unnormalised
    a = Kun / Kun.sum(dim=1, keepdim=True)
    V = -proj_tang(x, a @ x)
    # kinetic energy  (1/N^2) sum_ij e^{x_i.D x_j} ||V_i||^2 = mean_i ||V_i||^2 m_i
    Kfull = torch.exp(S)
    m = Kfull.mean(dim=1)                                     # mobility at each token
    ke = (V.pow(2).sum(-1) * m).mean().item()
    return V, ke


def token_azimuth(x):
    return torch.atan2(x[:, 1], x[:, 0])


def run(eps, tau=0.05, steps=6000, checks=300):
    torch.manual_seed(SEED)
    x = normalize(torch.randn(N, 3))
    a_prev = token_azimuth(x)
    swept = torch.zeros(N)
    every = max(1, steps // checks)
    ts, kin, ang = [], [], []
    for k in range(1, steps + 1):
        V, ke = field_and_kinetic_energy(x, eps)
        x = normalize(x + tau * V)
        a_now = token_azimuth(x)
        da = (a_now - a_prev + torch.pi) % (2 * torch.pi) - torch.pi
        swept += da
        a_prev = a_now
        if k % every == 0:
            ts.append(k * tau)
            kin.append(ke)
            ang.append(swept.mean().item())
    return np.array(ts), np.array(kin), np.array(ang), x


def main():
    eps_list = [0.0, 0.05, 0.10]
    colors = ["#7f7f7f", "#ff7f0e", "#1f77b4"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    for eps, c in zip(eps_list, colors):
        ts, kin, ang, xf = run(eps)
        lbl = rf"$\varepsilon={eps:.2f}$"
        axA.semilogy(ts, np.maximum(kin, 1e-16), color=c, lw=1.9, label=lbl)
        axB.plot(ts, ang, color=c, lw=1.9, label=lbl)
        print(f"eps={eps:.2f}  final kinetic energy={kin[-1]:.3e}  "
              f"final z_rms={xf[:,2].pow(2).mean().sqrt().item():.3e}  "
              f"mean rate(2nd half)={(ang[-1]-ang[len(ang)//2])/(ts[-1]-ts[len(ts)//2]):+.4f}")

    axA.set_xlabel("time $t$")
    axA.set_ylabel("mobility-weighted kinetic energy")
    axA.set_title("(a) symmetric flow comes to rest; asymmetric flow keeps moving")
    axA.legend(frameon=False, fontsize=9)

    axB.set_xlabel("time $t$")
    axB.set_ylabel(r"cumulative rotation $\theta(t)$ about $\omega$")
    axB.set_title("(b) persistent rotation, rate $\\propto\\varepsilon$")
    axB.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("saved", OUT)


if __name__ == "__main__":
    main()
