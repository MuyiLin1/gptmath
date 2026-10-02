"""
Planar invariance (Prop. "planar invariance") and what it implies for the tilt of
the ring (Corollary "geometry").

PART 1.  Is the invariance theorem true?
    Place all tokens exactly on a random great circle (random normal nu,
    random non-uniform density along the circle), with random eps, beta,
    and random wind axis omega.  Measure |nu . xdot| for every token.
    The proposition predicts machine zero at every order in eps.

PART 2.  What do the settled states from random starts look like?
    Run the flow from generic (non-planar) random initial data for 5 seeds and
    report, for the settled state,
      (a) planarity  = sigma_min / sigma_mid  of the position covariance
          (0 => exactly planar),
      (b) tilt angle = angle between the best-fit normal nu and e_3.
    Aligned A_sparse at eps = 0.3: the ring stays on the equator (tilt ~0.1 deg).
    Skewed A at eps = 0.7 (kappa = 2.1, far outside the ring regime): the tokens
    have coarsened into three equal clusters (the rotating triangle), so (a) is
    ~1e-16 trivially (three points are always coplanar) and (b) is the tilt of the
    triangle's plane.  This run does
    NOT measure the tilt of the rotating ring; for that, and for how the tilt
    depends on the type of start, see tilt_check.py.

PART 3.  Does the field tilt an exactly planar ring?
    Start on a great circle tilted 30 deg from e_3 (skewed A, eps = 0.7).  By
    Part 1 the normal cannot move while the ring is flat, so the tilt stays at
    exactly 30 deg; it changes only after round-off, amplified by the buckling
    instability, has made the ring non-planar.  This confirms Corollary
    "geometry": a formed flat ring is not tilted further; the tilt is imprinted
    while the ring forms.
"""
import numpy as np

# --- matrices from the paper's Experiments section ----------------------------
# A_SPARSE has dual axis -e3 (the transpose of the paper's A_sparse); only the sense
# of rotation differs, which affects neither planarity nor tilt.
A_SPARSE = np.array([[0., 1, 0], [-1, 0, 0], [0, 0, 0]])
A_SKEWED = np.array([[0., -3, 0], [3, 0, -3], [0, 3, 0]])


def flow(X, D):
    """Particle ODE right-hand side with V = -Id (paper eq. particle-ode)."""
    S = X @ D @ X.T
    S = S - S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = W @ X
    return -(F - np.sum(X * F, axis=1)[:, None] * X)


def A_from_axis(w):
    w = np.asarray(w, float)
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def great_circle(nu, angles):
    """N points on the great circle with unit normal nu, at given angles."""
    nu = nu / np.linalg.norm(nu)
    tmp = np.array([1.0, 0, 0])
    if abs(nu @ tmp) > 0.9:
        tmp = np.array([0.0, 1, 0])
    e1 = np.cross(nu, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nu, e1)
    return np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2, nu


def banner(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


# =============================================================================
banner("PART 1  invariance of great circles  (predict: machine zero)")
rng = np.random.default_rng(0)
worst = 0.0
print(f"  {'trial':>5} {'eps':>7} {'beta':>6} {'N':>5}   max |nu . xdot|")
for trial in range(8):
    N = int(rng.integers(20, 120))
    nu = rng.normal(size=3)
    # deliberately non-uniform (clumped) density along the circle
    angles = np.sort(rng.uniform(0, 2 * np.pi, N)) ** rng.uniform(0.7, 1.6)
    X, nu = great_circle(nu, angles)
    eps = rng.uniform(-1.5, 1.5)
    beta = rng.uniform(0.1, 5.0)
    omega = rng.normal(size=3)
    D = np.diag([0, 0, beta]) + eps * A_from_axis(omega)
    normal_vel = np.abs(flow(X, D) @ nu).max()
    worst = max(worst, normal_vel)
    print(f"  {trial:>5} {eps:>+7.3f} {beta:>6.3f} {N:>5}   {normal_vel:.3e}")
print(f"\n  worst deviation over all trials: {worst:.3e}")
print(f"  {'PASS -- great circles are exactly invariant' if worst < 1e-12 else 'FAIL'}")

# also confirm the paper's own two matrices, at several eps
print("\n  paper's matrices on the equator (nu = e_3):")
ang = np.linspace(0, 2 * np.pi, 200, endpoint=False)
Xeq = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
for name, A in [("sparse", A_SPARSE), ("skewed", A_SKEWED)]:
    for eps in [0.1, 0.3, 0.7, 2.0]:
        D = np.diag([0, 0, 1.0]) + eps * A
        v = np.abs(flow(Xeq, D) @ np.array([0, 0, 1.0])).max()
        print(f"    {name:>6}  eps={eps:>4}:  max |e3 . xdot| = {v:.3e}")


# =============================================================================
banner("PART 2  settled states from random starts (aligned eps=0.3; skewed eps=0.7, cluster regime)")


def settle(A, eps, seed, N=400, beta=1.0, dt=0.05, steps=6000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    D = np.diag([0, 0, beta]) + eps * A
    for _ in range(steps):
        X = X + dt * flow(X, D)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def diagnose(X):
    """Return (planarity, tilt_deg, normal) for a settled configuration."""
    C = X.T @ X / len(X)
    w, V = np.linalg.eigh(C)          # ascending
    planarity = w[0] / w[1]           # 0 => exactly planar
    nu = V[:, 0]                      # normal = least-variance direction
    if nu[2] < 0:
        nu = -nu
    tilt = np.degrees(np.arccos(np.clip(abs(nu[2]), 0, 1)))
    return planarity, tilt, nu


for name, A, eps in [("sparse (aligned)", A_SPARSE, 0.3),
                     ("skewed  (clusters)", A_SKEWED, 0.7)]:
    print(f"\n  {name},  eps = {eps}")
    print(f"    {'seed':>5} {'planarity':>12} {'tilt (deg)':>12}   normal")
    tilts = []
    for seed in [0, 1, 2, 3, 4]:
        X = settle(A, eps, seed)
        p, t, nu = diagnose(X)
        tilts.append(t)
        print(f"    {seed:>5} {p:>12.3e} {t:>12.4f}   "
              f"({nu[0]:+.3f}, {nu[1]:+.3f}, {nu[2]:+.3f})")
    tilts = np.array(tilts)
    print(f"    tilt spread across seeds: {tilts.max() - tilts.min():.4f} deg "
          f"(mean {tilts.mean():.4f})")

# check whether an EXACTLY planar tilted ring stays put: start on a 30-degree
# tilted great circle
banner("PART 3  does the field tilt an exactly planar ring?  (Corollary 'geometry')")
nu0 = np.array([np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))])
X, nu0 = great_circle(nu0, np.linspace(0, 2 * np.pi, 300, endpoint=False))
D = np.diag([0, 0, 1.0]) + 0.7 * A_SKEWED
print(f"  start: exactly planar ring, normal tilted 30 deg from e_3")
print(f"  {'step':>7} {'tilt (deg)':>12} {'planarity':>12}")
for s in range(4001):
    if s % 1000 == 0:
        p, t, nu = diagnose(X)
        print(f"  {s:>7} {t:>12.6f} {p:>12.3e}")
    X = X + 0.05 * flow(X, D)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
print("\n  Reading: while planarity stays ~1e-16 the tilt is exactly 30 deg -- the field")
print("  does not tilt a flat ring (Corollary 'geometry'). It changes only once the ring")
print("  buckles (planarity grows), i.e. through the transverse instability.")
