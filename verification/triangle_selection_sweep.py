"""
Long-time attractor of the self-attention flow (Proposition "triangle
selection", Remark on global status).

PURPOSE
  Numerical support for the claim that, from generic random starts, the tokens
  collapse onto three equal clusters that rotate rigidly around the equator.

WHAT IT DOES
  For a grid of (N, beta, eps) and several random seeds each, it runs the flow
  to long time and reports, for the settled state:
    - k        : number of DOMINANT clusters (groups holding >=10% of tokens)
    - sizes    : the cluster sizes (to see if they're roughly equal)
    - shape_drift : how much the (rotation-invariant) shape is still changing.
                    Near 0 => genuinely converged. If NOT small, the run was too
                    short and the k value is unreliable -- increase T.
    - spread   : angular spread; distinguishes a real ring/cluster state from a
                 single collapsed blob.

  It prints a per-run table and then a SUMMARY verdict.

HOW TO RUN
  python verification/triangle_selection_sweep.py
  (Takes a few minutes. To go faster, trim the grids at the bottom.)
"""
import numpy as np

# ---------------------------------------------------------------------
# The flow: true exponential softmax kernel, V = -Id, tangential projection.
# ---------------------------------------------------------------------
def flow(X, D):
    S = X @ D @ X.T
    S = S - S.max(axis=1, keepdims=True)
    W = np.exp(S); W /= W.sum(axis=1, keepdims=True)
    F = -(W @ X)                                   # V = -Id
    return F - np.sum(X * F, axis=1)[:, None] * X  # tangential projection

def A_from_axis(w):
    w = np.asarray(w, float)
    return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

# ---------------------------------------------------------------------
# Integrate from a random start; track rotation-invariant shape convergence.
# ---------------------------------------------------------------------
def settle(N, beta, eps, seed, T, dt=0.05, omega=(0,0,1)):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N,3)); X /= np.linalg.norm(X,axis=1,keepdims=True)
    D = np.diag([0,0,beta]) + eps*A_from_axis(omega)
    prev_shape = None
    shape_drift = np.nan
    for s in range(int(T/dt)):
        X = X + dt*flow(X,D)
        X /= np.linalg.norm(X,axis=1,keepdims=True)
        if s % 400 == 0:
            # sorted pairwise distances = a rotation/permutation-invariant fingerprint
            d = np.sort(np.linalg.norm(X[:,None]-X[None,:],axis=2).ravel())
            if prev_shape is not None:
                shape_drift = np.abs(d - prev_shape).mean()
            prev_shape = d
    return X, shape_drift

# ---------------------------------------------------------------------
# Diagnostics on the settled state.
# ---------------------------------------------------------------------
def analyze(X, tol=0.15, min_frac=0.10):
    N = len(X)
    used = np.zeros(N, bool); sizes = []
    for i in range(N):
        if used[i]: continue
        grp = np.linalg.norm(X-X[i],axis=1) < tol
        used |= grp; sizes.append(int(grp.sum()))
    sizes.sort(reverse=True)
    dominant = [s for s in sizes if s >= min_frac*N]
    k = len(dominant)
    # equality of the dominant clusters (max deviation from equal share)
    if k > 0:
        equal_dev = max(abs(s - N/k) for s in dominant) / (N/k)
    else:
        equal_dev = np.nan
    # is it a genuine multi-cluster state, or one collapsed blob?
    spread = np.linalg.norm(X - X.mean(0), axis=1).mean()
    return k, sizes, equal_dev, spread

# ---------------------------------------------------------------------
def main():
    print("="*78)
    print("Long-time attractor of the self-attention flow")
    print("="*78)
    print("Reading each row: k = # dominant clusters; sizes should be ~equal;")
    print("shape_drift near 0 = converged (trust k); if large, increase T.\n")

    # ---- MAIN GRID: vary eps, several seeds, fixed N,beta -----------------
    print(f"{'N':>4}{'beta':>6}{'eps':>6}{'seed':>6}{'k':>4}"
          f"{'sizes':>18}{'equal_dev':>11}{'shape_drift':>13}{'spread':>9}")
    verdict_counts = {}
    for eps in [0.1, 0.3, 0.6, 1.0]:
        T = 8000 if eps <= 0.1 else (6000 if eps <= 0.3 else 4000)  # slow modes ~1/eps^2
        for seed in range(4):
            X, drift = settle(60, 1.0, eps, seed, T=T)
            k, sizes, edev, spread = analyze(X)
            conv = "ok" if (drift < 1e-4) else "SHORT?"
            print(f"{60:>4}{1.0:>6}{eps:>6}{seed:>6}{k:>4}"
                  f"{str(sizes[:5]):>18}{edev:>11.2f}{drift:>13.2e}{spread:>9.3f}  {conv}")
            if drift < 1e-4:
                verdict_counts[k] = verdict_counts.get(k,0)+1

    # ---- ROBUSTNESS: does k depend on N or beta? -------------------------
    print("\nRobustness (eps=0.6, T=4000):")
    print(f"{'N':>4}{'beta':>6}{'seed':>6}{'k':>4}{'sizes':>18}{'shape_drift':>13}")
    for (N,beta) in [(30,1.0),(90,1.0),(60,0.5),(60,2.0)]:
        for seed in [0,1]:
            X,drift = settle(N,beta,0.6,seed,T=4000)
            k,sizes,edev,spread = analyze(X)
            print(f"{N:>4}{beta:>6}{seed:>6}{k:>4}{str(sizes[:5]):>18}{drift:>13.2e}")

    # ---- SUMMARY VERDICT --------------------------------------------------
    print("\n" + "="*78)
    print("SUMMARY (converged runs only):")
    for k in sorted(verdict_counts):
        print(f"  {verdict_counts[k]} runs settled to k = {k} dominant clusters")
    total = sum(verdict_counts.values())
    if verdict_counts.get(3,0) == total and total > 0:
        print("\n  VERDICT: every converged run gave EXACTLY 3 equal clusters.")
        print("  The three-cluster attractor is robust.")
    elif verdict_counts.get(3,0) >= 0.7*total and total > 0:
        print("\n  VERDICT: mostly 3 clusters, some exceptions. The phenomenon is")
        print("  real but not universal -- investigate the exceptions before")
        print("  claiming 'always 3'. Note which (eps,seed) differed.")
    else:
        print("\n  VERDICT: k is NOT robustly 3. The clean three-cluster story does")
        print("  NOT hold for these parameters (see the distribution of k above).")
    print("="*78)

if __name__ == "__main__":
    main()