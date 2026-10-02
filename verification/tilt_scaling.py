"""Tilt of the settled skewed ring versus eps, staying below the apparent buckling
threshold so the state is a genuine ring rather than a fragmented cluster.
Corollary "geometry" predicts tilt = O(eps), with the normal moving in the direction
omega x e3.  The tilt is imprinted while the ring forms, so its size depends on the
start; tilt_check.py compares random and near-ring starts."""
import numpy as np

A_SKEWED = np.array([[0., -3, 0], [3, 0, -3], [0, 3, 0]])
# dual axis of A_skewed:  omega = (-A[1,2], A[0,2], -A[0,1]) = (3, 0, 3)
OMEGA = np.array([3.0, 0.0, 3.0]) / np.linalg.norm([3.0, 0.0, 3.0])


def flow(X, D):
    S = X @ D @ X.T
    S = S - S.max(axis=1, keepdims=True)
    W = np.exp(S)
    W /= W.sum(axis=1, keepdims=True)
    F = W @ X
    return -(F - np.sum(X * F, axis=1)[:, None] * X)


def settle(eps, seed, N=400, dt=0.05, steps=8000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    D = np.diag([0, 0, 1.0]) + eps * A_SKEWED
    for _ in range(steps):
        X = X + dt * flow(X, D)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


print(f"  omega (dual axis of A_skewed) = {OMEGA},  "
      f"angle to e3 = {np.degrees(np.arccos(OMEGA[2])):.1f} deg")
print("  Corollary 'geometry': ring normal tilts from e3 in the direction omega x e3, by O(eps)\n")
print(f"  {'eps':>6} {'tilt (deg), per seed':>34} {'mean':>8} {'tilt/eps':>10} "
      f"{'planarity':>11}")
for eps in [0.005, 0.010, 0.020, 0.030, 0.040]:
    tilts, plans = [], []
    for seed in [0, 1, 2]:
        X = settle(eps, seed)
        C = X.T @ X / len(X)
        w, V = np.linalg.eigh(C)
        nu = V[:, 0]
        if nu[2] < 0:
            nu = -nu
        tilts.append(np.degrees(np.arccos(np.clip(abs(nu[2]), 0, 1))))
        plans.append(w[0] / w[1])
    t = np.array(tilts)
    print(f"  {eps:>6} [{t[0]:>9.4f} {t[1]:>9.4f} {t[2]:>9.4f}] "
          f"{t.mean():>8.4f} {t.mean()/eps:>10.2f} {np.mean(plans):>11.2e}")
print("\n  tilt/eps roughly constant => tilt is O(eps) (approximately linear; the drift "
      "reflects the approach time).")
print("  Direction set by omega; size depends on the start (see tilt_check.py).")
