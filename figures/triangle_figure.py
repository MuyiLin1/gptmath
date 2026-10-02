"""Generate triangle.png: two-panel figure showing the ring → triangle coarsening.

Left panel  – the transient rotating ring  (early time, T_ring steps)
Right panel – the settled rotating triangle (long time, T_tri steps)

Both run the same sparse-A flow so the ring and triangle are from identical
initial conditions, differing only in how long we integrate.

Run from export/:  PYTHONPATH=. python figures/triangle_figure.py
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

from tfdy.optim_diracs import usa_flow, proj, proj_tang
from tfdy.plotting import PlotConf3D

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'figures')

D_DIM = 3
N     = 60       # fewer tokens so clusters are visually distinct
BETA  = 1.0
EPS   = 0.6      # large enough that the slow O(eps^2) ring instability completes in the window
SEED  = 0
TAU   = 0.05
SGN   = 1

Ds = torch.diag(torch.tensor([0.0, 0.0, 1.0]))
Id = torch.eye(D_DIM)
# dual axis -e3: the transpose of the paper's A_sparse; only the sense of rotation differs
A_sparse = torch.tensor([[0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]])

T_RING = 800   # m3 amplitude ~0.73 at this point: three lumps clearly forming but not yet collapsed
T_TRI  = 6000  # steps to reach the settled rotating triangle


def flow_step(x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    K = torch.exp(BETA * (x @ (D @ x.T)))
    K = K / K.sum(dim=1, keepdim=True)
    update = K @ x @ Id.T
    return -SGN * proj_tang(x, update)


def draw_equator(ax, color='0.55', lw=1.2, ls='--', alpha=0.7):
    """Faint dashed true e_3-equator as a fixed reference."""
    t = torch.linspace(0, 2 * torch.pi, 300)
    ax.plot(torch.cos(t).numpy(), torch.sin(t).numpy(),
            torch.zeros(300).numpy(),
            color=color, linewidth=lw, linestyle=ls, alpha=alpha, zorder=2)


def draw_guide_ring(ax, normal: torch.Tensor, color='0.35', lw=1.6, alpha=0.7):
    """Solid gray guide ring perpendicular to normal."""
    normal = normal / normal.norm()
    ref = torch.tensor([1.0, 0.0, 0.0])
    if abs(torch.dot(ref, normal)) > 0.9:
        ref = torch.tensor([0.0, 1.0, 0.0])
    u = ref - torch.dot(ref, normal) * normal
    u = u / u.norm()
    v = torch.linalg.cross(normal, u)
    t = torch.linspace(0, 2 * torch.pi, 300)
    ring = torch.outer(torch.cos(t), u) + torch.outer(torch.sin(t), v)
    ax.plot(ring[:, 0].numpy(), ring[:, 1].numpy(), ring[:, 2].numpy(),
            color=color, linewidth=lw, alpha=alpha, zorder=2)


def velocity_field(x: torch.Tensor) -> torch.Tensor:
    D = Ds + EPS * A_sparse
    return flow_step(x, D)


def render_panel(ax, x: torch.Tensor, show_vel: bool = True,
                 guide_normal: torch.Tensor = None,
                 show_equator_ref: bool = False):
    PC = PlotConf3D()
    PC.fig = ax.figure
    PC.axs = [ax]
    PC.plot_sphere(idx=0, color=(0.2745, 0.2549, 0.5882), alpha=0.25,
                   edgecolor='none')

    if guide_normal is not None:
        draw_guide_ring(ax, guide_normal)
    if show_equator_ref:
        draw_equator(ax)

    if show_vel:
        vel = velocity_field(x)
        speed = vel.norm(dim=-1)
        scale = 0.45 / (speed.max() + 1e-9)
        v = vel * scale
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=40, color='orange',
                   depthshade=False, zorder=3)
        ax.quiver(x[:, 0], x[:, 1], x[:, 2],
                  v[:, 0], v[:, 1], v[:, 2],
                  color='crimson', linewidth=1.2, arrow_length_ratio=0.4,
                  zorder=4)
    else:
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=40, color='orange',
                   depthshade=False, zorder=3)


def main():
    torch.manual_seed(SEED)
    x = proj(torch.randn(N, D_DIM))
    D = Ds + EPS * A_sparse

    # integrate to ring snapshot
    for _ in range(T_RING):
        x = x + TAU * flow_step(x, D)
        x = x / x.norm(dim=-1, keepdim=True)
    x_ring = x.clone()

    # continue to triangle
    for _ in range(T_TRI - T_RING):
        x = x + TAU * flow_step(x, D)
        x = x / x.norm(dim=-1, keepdim=True)
    x_tri = x.clone()

    # equatorial guide normal for ring panel (tokens are near-equatorial)
    ring_normal = torch.tensor([0.0, 0.0, 1.0])

    # triangle guide normal: best-fit plane of the 3 cluster centroids
    xnp = x_tri.numpy()
    # find cluster centres (3 groups by k-means-like greedy assignment)
    centres = [xnp[0]]
    for p in xnp[1:]:
        if all(np.linalg.norm(p - c) > 0.5 for c in centres):
            centres.append(p)
        if len(centres) == 3:
            break
    C = np.array(centres[:3] if len(centres) >= 3 else [xnp[0], xnp[N//3], xnp[2*N//3]])
    # normal to the plane of the 3 centres
    tri_normal = torch.tensor(
        np.cross(C[1] - C[0], C[2] - C[0]), dtype=torch.float32)
    tri_normal = tri_normal / tri_normal.norm()
    if tri_normal[2] < 0:
        tri_normal = -tri_normal

    # --- figure -----------------------------------------------------------------
    fig = plt.figure(figsize=(9, 4.5))
    SPHERE_COLOR = (0.2745, 0.2549, 0.5882)
    ELEV, AZIM = 22, -55

    for col, (x_snap, title, guide, eq_ref) in enumerate([
        (x_ring, '(a) Rotating ring\n(three groups forming)',
         ring_normal, False),
        (x_tri,  '(b) Settled attractor:\nrotating triangle',
         tri_normal, True),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')
        ax.view_init(elev=ELEV, azim=AZIM)

        # sphere surface
        u = torch.linspace(0, 2 * torch.pi, 40)
        v = torch.linspace(0, torch.pi, 40)
        xs = torch.outer(torch.cos(u), torch.sin(v))
        ys = torch.outer(torch.sin(u), torch.sin(v))
        zs = torch.outer(torch.ones_like(u), torch.cos(v))
        ax.plot_surface(xs.numpy(), ys.numpy(), zs.numpy(),
                        color=SPHERE_COLOR, alpha=0.18, edgecolor='none',
                        linewidth=0)

        draw_guide_ring(ax, guide)
        if eq_ref:         # dashed true equator reference in triangle panel
            draw_equator(ax)

        vel = velocity_field(x_snap)
        speed = vel.norm(dim=-1)
        scale = 0.45 / (speed.max() + 1e-9)
        v_arr = (vel * scale).numpy()
        xn = x_snap.numpy()

        ax.scatter(xn[:, 0], xn[:, 1], xn[:, 2], s=40, color='orange',
                   depthshade=False, zorder=3)
        ax.quiver(xn[:, 0], xn[:, 1], xn[:, 2],
                  v_arr[:, 0], v_arr[:, 1], v_arr[:, 2],
                  color='crimson', linewidth=1.0, arrow_length_ratio=0.35,
                  zorder=4)

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        for pre in ('x', 'y', 'z'):
            getattr(ax, f'set_{pre}ticks')([])
            getattr(ax, f'set_{pre}ticklabels')('')
        for i, lbl in enumerate(['$z_1$', '$z_2$', '$z_3$']):
            getattr(ax, f'set_{"xyz"[i]}label')(lbl, fontsize=18, labelpad=-10)
        ax.set_title(title, fontsize=12, pad=6)
        ax.minorticks_off()

    fig.tight_layout(pad=0.5)
    out = os.path.join(OUT_DIR, 'triangle.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=600)
    print(f'wrote {out}')

    # sanity: count distinct clusters in triangle panel
    xnp = x_tri.numpy()
    keep = [xnp[0]]
    for p in xnp[1:]:
        if min(np.linalg.norm(p - q) for q in keep) > 0.3:
            keep.append(p)
    print(f'distinct cluster positions in triangle panel: {len(keep)}  (expect 3)')


if __name__ == '__main__':
    main()
