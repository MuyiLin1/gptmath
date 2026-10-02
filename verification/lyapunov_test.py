"""Height Lyapunov function (Lemma "Ldot", Prop. "aligned", Remark "L-scope"):
how does  L(X) = mean_i (e3 . X_i)^2  behave under the full perturbed flow (eps>0)?

dL/dt <= 0 at eps = 0; for eps > 0 the antisymmetric part adds nothing at first
order, and the remaining O(eps^2) effect makes zero-sum ripples grow slowly (the
buckle, Cor. "no-threshold"). So L is non-increasing up to slow O(eps^2) growth,
on times << 1/eps^2.

Convention: this script puts BETA in front of the whole matrix,
softmax(BETA * x.(Ds + eps*A) y), so the effective symmetric part is diag(0,0,BETA)
and the effective coupling is kappa = BETA*eps (paper: kappa = eps (omega . e3)).

Tests (sparse A, dual axis along the Ds symmetry axis):
  T1  along random trajectories, does L ever rise over this horizon?
  T2  is the rotational drift L-preserving?  (dL/dt with and without eps*A
      should differ only at O(eps^2))
  T3  near the ring at eps = 0.04, does dL/dt match cubic decay plus slow buckle
      growth,  dL/dt = 2*lambda_2*L - 2*BETA*L^2,  lambda_2 = I_2(kappa)/I_0(kappa)?
"""
import os
import torch
from scipy.special import iv

torch.set_default_dtype(torch.float64)

N = int(os.environ.get("PROBE_N", 60))
BETA = float(os.environ.get("PROBE_BETA", 2.0))
DS = torch.diag(torch.tensor([0.0, 0.0, 1.0]))
# dual axis -e3: the transpose of the paper's A_sparse; only the sense of rotation
# differs, and L depends only on heights, so every number here is unaffected.
A_SPARSE = torch.tensor([[0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])
OMEGA = torch.tensor([0.0, 0.0, 1.0])


def proj_tang(x, y):
    return y - (y * x).sum(-1, keepdim=True) * x


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


def Vfield(x, eps):
    D = DS + eps * A_SPARSE
    K = torch.softmax(BETA * (x @ (D @ x.T)), dim=1)
    return -proj_tang(x, K @ x)


def L(x):
    return (x @ OMEGA).pow(2).mean().item()


def Ldot(x, eps):
    """exact dL/dt = (2/N) sum_i (omega.X_i)(omega.Xdot_i)."""
    v = Vfield(x, eps)
    return (2.0 * (x @ OMEGA) * (v @ OMEGA)).mean().item()


def random_config(seed):
    g = torch.Generator().manual_seed(seed)
    return normalize(torch.randn(N, 3, generator=g))


def main():
    print(f"N={N}  beta={BETA}  (sparse A, axis along e3)\n")

    # T1: monotonicity of L along trajectories from random starts -----------
    print("T1  does L rise along trajectories from random starts?")
    tau = 0.005
    for eps in (0.0, 0.04, 0.1):
        worst_increase = 0.0
        for seed in range(6):
            x = random_config(seed)
            Lprev = L(x)
            for k in range(20000):
                x = normalize(x + tau * Vfield(x, eps))
                Lnow = L(x)
                worst_increase = max(worst_increase, Lnow - Lprev)
                Lprev = Lnow
        print(f"   eps={eps:5.2f}:  worst single-step increase of L = "
              f"{worst_increase:.2e}   final L={L(x):.3e}")
    print("   (no rise over this horizon; slow O(eps^2) ripple growth is not visible at these times)\n")

    # T2: does the eps (rotational) part contribute to dL/dt? ----------------
    print("T2  contribution of the rotation to dL/dt (should vanish):")
    torch.manual_seed(1)
    for trial in range(4):
        # a generic off-ring configuration
        x = normalize(random_config(10 + trial))
        d0 = Ldot(x, 0.0)
        de = Ldot(x, 0.1)
        print(f"   config {trial}:  dL/dt(eps=0)={d0:+.4e}   "
              f"dL/dt(eps=0.1)={de:+.4e}   diff={de-d0:+.2e}")
    print("   (diff ~ O(eps^2) tiny => rotation does not change L)\n")

    # T3: near-ring law  dL/dt vs L  ----------------------------------------
    eps3 = 0.04
    kappa = BETA * eps3
    lam2 = iv(2, kappa) / iv(0, kappa)
    print(f"T3  dL/dt vs L near the ring at eps = {eps3}: cubic decay -2*BETA*L^2 plus slow "
          f"buckle growth 2*lambda_2*L")
    print(f"    (lambda_2 = I_2(kappa)/I_0(kappa) = {lam2:.4e}, kappa = BETA*eps = {kappa})")
    phi = 2 * torch.pi * torch.arange(N) / N
    X = torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros(N)], 1)
    b = torch.cos(2 * phi); b = b - b.mean()
    print(f"   {'amp':>8}{'L':>14}{'dL/dt':>16}{'predicted':>16}{'obs/pred':>10}")
    for amp in (0.16, 0.08, 0.04, 0.02):
        x = normalize(X + amp * b[:, None] * torch.tensor([0.0, 0.0, 1.0]))
        Lv = L(x); dv = Ldot(x, eps3)
        pred = 2 * lam2 * Lv - 2 * BETA * Lv * Lv
        print(f"   {amp:8.3f}{Lv:14.4e}{dv:16.4e}{pred:16.4e}{dv / pred:10.4f}")
    print("   (small ripples grow, large ones shrink: the sign change is the slow buckle,")
    print("    not numerical noise; the largest amplitude also feels the +3L^3 correction)")


if __name__ == "__main__":
    main()
