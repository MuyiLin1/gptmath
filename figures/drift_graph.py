"""Regenerate the drift line-graph (driftGraph.jpg) for the paper.

It reproduces the comparison referenced at paper.tex (fig:drift): the drift
induced by the *true exponential* kernel versus the *first-order Taylor* kernel,
as a function of the perturbation strength epsilon.

Drift here is the mean tangential speed ||V_eps|| of the velocity field evaluated
at the *symmetric* (eps=0) steady-state ring -- i.e. how fast the antisymmetric
"ghost force" pushes the tokens around the ring. To leading order this equals
eps * ||V_1||, so:

  * at small eps both curves grow linearly with the same slope (set by ||V_1||);
  * for larger eps the true exponential bends away from the straight Taylor line
    (higher-order terms), and the Taylor kernel eventually produces unphysical
    negative attention weights (marked on the plot) -- it is a device for reading
    off the O(eps) coefficient, not a dynamical system in its own right.

Run from export/:  PYTHONPATH=. python figures/drift_graph.py
The figure is written to results/figures/.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

from tfdy.optim_diracs import usa_flow, proj, proj_tang

# --- model configuration (matches limit_cycle_figures.py) --------------------
D_DIM = 3
N = 400
TAU = 0.1
MAX_IT = 4000
BETA = 1.0
SGN = 1
SEED = 42

Ds = torch.diag(torch.tensor([0.0, 0.0, 1.0]))
Id = torch.eye(D_DIM)
# dual axis -e3: the transpose of the paper's A_sparse; only the sense of rotation differs
A_sparse = torch.tensor([[0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures')


def symmetric_steady_state():
    """The eps=0 reference ring (true-exponential, V=-Id flow)."""
    torch.manual_seed(SEED)
    x0 = proj(torch.randn(N, D_DIM))
    x, _ = usa_flow(
        Ds, A=None, epsilon=0.0, value=Id, normalize=True,
        n=N, tau=TAU, max_it=MAX_IT, beta=BETA, sigma=0.0, sgn=SGN, x=x0,
    )
    return x


def drift_speed(x, A, epsilon, kernel='true'):
    """Mean tangential speed of the velocity field at ``x``.

    Returns (mean_speed, min_attention_weight). ``min_attention_weight`` < 0
    flags the Taylor kernel leaving the physical (softmax) regime.
    """
    if kernel == 'true':
        D_full = Ds + epsilon * A
        K = torch.exp(BETA * (x @ (D_full @ x.T)))
    elif kernel == 'taylor':
        K0 = torch.exp(BETA * (x @ (Ds @ x.T)))
        K = K0 * (1.0 + BETA * epsilon * (x @ (A @ x.T)))
    else:
        raise ValueError(kernel)
    min_w = K.min().item()
    K = K / K.sum(dim=1, keepdim=True)
    update = K @ x @ Id.T
    vel = proj_tang(x, update)
    speed = torch.linalg.vector_norm(vel, dim=-1).mean().item()
    return speed, min_w


def main():
    x0 = symmetric_steady_state()

    eps_vals = np.linspace(0.0, 1.2, 31)
    true_d, tay_d = [], []
    tay_neg_eps = None
    for eps in eps_vals:
        st, _ = drift_speed(x0, A_sparse, float(eps), 'true')
        sy, minw = drift_speed(x0, A_sparse, float(eps), 'taylor')
        true_d.append(st)
        tay_d.append(sy)
        if tay_neg_eps is None and minw < 0 and eps > 0:
            tay_neg_eps = float(eps)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(eps_vals, true_d, '-o', color='C0', ms=3.5, lw=1.9,
            label=r'true exponential  $e^{\,x^\top(D^s+\varepsilon A)y}$')
    ax.plot(eps_vals, tay_d, '-s', color='C3', ms=3.5, lw=1.9,
            label=r'first-order Taylor  $e^{\,x^\top D^s y}(1+\varepsilon\, x^\top Ay)$')

    if tay_neg_eps is not None:
        ax.axvline(tay_neg_eps, color='0.55', ls='--', lw=1.2)
        ymax = max(max(true_d), max(tay_d))
        ax.text(tay_neg_eps + 0.01, 0.03 * ymax,
                'Taylor weights\nturn negative',
                color='0.4', fontsize=8, ha='left', va='bottom')

    ax.set_xlabel(r'perturbation strength  $\varepsilon$', fontsize=11)
    ax.set_ylabel(r'mean drift speed  $\|\mathcal{V}_\varepsilon\|$'
                  '\n(at the symmetric steady-state ring)', fontsize=11)
    ax.set_title(r'Drift vs. $\varepsilon$: true exponential vs. first-order Taylor',
                 fontsize=12)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.margins(x=0.01)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, 'driftGraph.jpg')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200)
    print('wrote', out)
    print('taylor negative-weight onset eps =', tay_neg_eps)


if __name__ == '__main__':
    main()
