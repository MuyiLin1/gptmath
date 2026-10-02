"""Regenerate the limit-cycle figures for the paper from a single, standardized
flow: the *true exponential* kernel with value matrix V = -Id (the model of
Assumption "value matrix" / eq:particle-ode; implemented as value=Id with sgn=+1).

Four figures, all from the same model, differing only by the perturbation:

    symmetric.png : eps = 0            -> static equatorial ring (no motion)
    sparse.jpg    : sparse A           -> rotating equatorial ring (axis z)
    skewed.jpg    : skewed A           -> rotating tilted ring

To make "frozen vs. spinning" legible in a still image, the perturbed figures
overlay tangent velocity arrows (the instantaneous field) on the tokens.

Run from export/:  PYTHONPATH=. python figures/limit_cycle_figures.py
Figures are written to results/figures/.
"""
import os
import torch
import matplotlib
matplotlib.use('Agg')          # headless rendering
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False   # no system LaTeX needed; use mathtext

from tfdy.optim_diracs import usa_flow, proj, proj_tang
from tfdy.plotting import PlotConf3D

# --- model configuration (matches the paper) ---------------------------------
D_DIM = 3
N = 400
TAU = 0.1
MAX_IT = 4000
BETA = 1.0
SGN = 1                      # +1 => tokens settle on the equator (verified)
SEED = 42

Ds = torch.diag(torch.tensor([0.0, 0.0, 1.0]))      # symmetric part diag(0,0,1)
Id = torch.eye(D_DIM)                               # value=Id with sgn=+1, i.e. V = -Id

# dual axis -e3: the transpose of the paper's A_sparse; only the sense of rotation differs
A_sparse = torch.tensor([[0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])
# Skewed A whose dual axis omega = (-A[1,2], A[0,2], -A[0,1]) = (3, 0, 3):
# a 45-degree tilt away from z, so the limit-cycle ring is visibly tilted.
A_skewed = torch.tensor([[0.0, -3.0, 0.0],
                         [3.0, 0.0, -3.0],
                         [0.0, 3.0, 0.0]])
# Both panels are rendered BELOW their apparent (finite-window) buckling thresholds
# so the tokens form a genuine, well-populated ring rather than a buckled/collapsed
# set of a few clusters. The buckle grows at every eps != 0, but slowly (rate ~kappa^2/8);
# over the simulated window it looks like a threshold at eps_c ~ 0.15 (aligned) and
# ~ 0.05 (skewed: three times larger omega . e3, so it buckles at ~3x smaller eps).
# Well above that the ring fragments to a handful of points, which -- being
# trivially coplanar -- would look deceptively like a clean flat ring.
EPS = 0.1
# The skewed panel is rendered just below its apparent buckling threshold so the
# (honest) O(eps) tilt is still visible: at this strength the intact ring settles ~4-5 deg
# off the equator -- plainly distinct from the equatorial sparse panel, yet far
# short of the full 45 deg plane perp-omega (reached only as the asymmetry
# -> infinity, well past the point where no ring survives).
EPS_SKEWED = 0.04

# Repo root = parent of this file's directory.
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures')


def run_flow(A, epsilon):
    """Run the standardized true-exponential, V=-Id flow to its steady state."""
    torch.manual_seed(SEED)
    x0 = proj(torch.randn(N, D_DIM))
    x, _ = usa_flow(
        Ds, A=A, epsilon=epsilon, value=Id, normalize=True,
        n=N, tau=TAU, max_it=MAX_IT, beta=BETA, sigma=0.0, sgn=SGN, x=x0,
    )
    return x


def velocity_field(x, A, epsilon):
    """Instantaneous tangent velocity of the flow at configuration ``x``."""
    D_full = Ds if (A is None or epsilon == 0.0) else Ds + epsilon * A
    K = torch.exp(BETA * (x @ (D_full @ x.T)))
    K = K / K.sum(dim=1, keepdim=True)
    update = K @ x @ Id.T
    # usa_flow steps x <- x - sgn*tau*proj_tang(x, update); velocity is that
    # increment direction (per unit tau).
    return -SGN * proj_tang(x, update)


def dual_axis(A):
    """Axis omega dual to the antisymmetric matrix A (so that A x = omega x x).

    For the symmetric baseline (A is None) the relevant symmetry axis of
    Ds=diag(0,0,1) is the z-axis.
    """
    if A is None:
        return torch.tensor([0.0, 0.0, 1.0])
    omega = torch.tensor([-A[1, 2], A[0, 2], -A[0, 1]])
    return omega / torch.linalg.vector_norm(omega)


def token_ring_normal(x):
    """Normal of the great circle the tokens actually settle on.

    This is the smallest-variance principal axis of the (centered) token
    cloud, i.e. the direction in which the ring is thinnest. Unlike
    ``dual_axis``, it reflects where the orange tokens really are rather than
    the analytic axis omega -- so the guide ring drawn from it honestly traces
    the (small) tilt of the limit cycle.
    """
    xc = x - x.mean(dim=0, keepdim=True)
    cov = xc.T @ xc
    evals, evecs = torch.linalg.eigh(cov)
    n = evecs[:, 0]
    return n / torch.linalg.vector_norm(n)


def draw_guide_ring(ax, omega, n=400, **kwargs):
    """Draw the great circle perpendicular to ``omega`` as a faint guide curve."""
    omega = omega / torch.linalg.vector_norm(omega)
    # Two orthonormal vectors spanning the plane perpendicular to omega.
    ref = torch.tensor([1.0, 0.0, 0.0])
    if torch.abs(torch.dot(ref, omega)) > 0.9:
        ref = torch.tensor([0.0, 1.0, 0.0])
    u = ref - torch.dot(ref, omega) * omega
    u = u / torch.linalg.vector_norm(u)
    v = torch.linalg.cross(omega, u)
    t = torch.linspace(0, 2 * torch.pi, n)
    ring = torch.outer(torch.cos(t), u) + torch.outer(torch.sin(t), v)
    dkwargs = {'color': '0.35', 'linewidth': 1.6, 'alpha': 0.7, 'zorder': 2}
    dkwargs.update(kwargs)
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], **dkwargs)


def draw_equator_ref(ax):
    """Solid colored true e_3-equator reference so the tilt in the skewed panel is clear."""
    t = torch.linspace(0, 2 * torch.pi, 300)
    ax.plot(torch.cos(t).numpy(), torch.sin(t).numpy(), torch.zeros(300).numpy(),
            color='#2166ac', linewidth=2.2, linestyle='-', alpha=0.85, zorder=3)


def save_fig(x, fname, A=None, epsilon=0.0, show_motion=False, elev=20, azim=-60,
             guide_normal=None, show_equator_ref=False):
    """Render tokens on the sphere; optionally overlay velocity arrows.

    If ``guide_normal`` is given, the faint gray guide ring is the great circle
    perpendicular to it (used to trace the *actual* token ring); otherwise the
    guide ring is the analytic circle perpendicular to the axis dual to ``A``.
    """
    PC = PlotConf3D()
    PC.init_axs(1, labelpad=-10, show_ticks=False, axlabelfontsize=25)
    ax = PC.axs[0]
    ax.view_init(elev=elev, azim=azim)

    PC.plot_sphere(idx=0, color=(0.2745, 0.2549, 0.5882), alpha=0.25,
                   edgecolor='none')

    # Faint guide ring: the great circle the limit cycle lives on.
    normal = dual_axis(A) if guide_normal is None else guide_normal
    draw_guide_ring(ax, normal)
    if show_equator_ref:
        draw_equator_ref(ax)

    if show_motion:
        vel = velocity_field(x, A, epsilon)
        speed = torch.linalg.vector_norm(vel, dim=-1)
        # Scale arrows for visibility.
        scale = 0.35 / (speed.max() + 1e-9)
        v = vel * scale
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=16, color='orange',
                   depthshade=False, zorder=3)
        ax.quiver(x[:, 0], x[:, 1], x[:, 2],
                  v[:, 0], v[:, 1], v[:, 2],
                  color='crimson', linewidth=1.0, arrow_length_ratio=0.4,
                  zorder=4)
    else:
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=18, color='orange',
                   depthshade=False, zorder=3)

    out = os.path.join(OUT_DIR, fname)
    PC.fig.tight_layout(pad=0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    PC.fig.savefig(out, dpi=600)
    print(f'wrote {out}')


def main():
    # Symmetric baseline: static equatorial ring (no motion arrows).
    x_sym = run_flow(A=None, epsilon=0.0)
    save_fig(x_sym, 'symmetric.png', show_motion=False)

    # 2) Perturbed (sparse A): rotating equatorial ring.
    x_sparse = run_flow(A=A_sparse, epsilon=EPS)
    save_fig(x_sparse, 'sparse.jpg', A=A_sparse, epsilon=EPS,
             show_motion=True)

    # 3) Skewed A: rotating tilted ring. The guide ring traces where the tokens
    #    actually settle (their thinnest principal axis), not the analytic axis
    #    perp-omega -- the true tilt is only O(eps), much less than the 45 deg of
    #    omega, so drawing perp-omega would overstate the skew. Rendered at
    #    EPS_SKEWED so the (honest) tilt is large enough to see.
    x_skewed = run_flow(A=A_skewed, epsilon=EPS_SKEWED)
    # lower elev makes the gap between the blue equator and the tilted gray ring visible
    save_fig(x_skewed, 'skewed.jpg', A=A_skewed, epsilon=EPS_SKEWED,
             show_motion=True, guide_normal=token_ring_normal(x_skewed),
             elev=10, azim=-60,
             show_equator_ref=True)

    # Quick numeric sanity report.
    for name, x, A, eps in [
        ('symmetric', x_sym, None, 0.0),
        ('sparse', x_sparse, A_sparse, EPS),
        ('skewed', x_skewed, A_skewed, EPS_SKEWED),
    ]:
        vel = velocity_field(x, A, eps)
        speed = torch.linalg.vector_norm(vel, dim=-1).mean().item()
        print(f'{name:9s} mean|z|={x[:, 2].abs().mean():.3f} '
              f'mean_speed={speed:.4f}')


if __name__ == '__main__':
    main()
