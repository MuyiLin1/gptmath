# Code for "Antisymmetry Breaks the Gradient-Flow Structure of Self-Attention"

This folder has all the code behind the figures and numerical checks in the paper.
We simulate the **true** (un-truncated, full-exponential) self-attention flow of
`N` tokens on the sphere `S^2`,

    Xdot_i = P_{X_i}( sum_j a_ij V X_j ),   a_ij = softmax_j( X_i . D X_j ),   D = D_s + eps * A,

with `V = -Id`, `D_s = diag(0, 0, 1)` (the ring experiments; `plot_isotropic.py` uses
`D_s = I`), and one of the two antisymmetric matrices from the Experiments section:

    A_sparse = [[0,-1,0],[1,0,0],[0,0,0]]     (dual axis omega = e3)
    A_skewed = [[0,-3,0],[3,0,-3],[0,3,0]]    (dual axis omega = (3,0,3), 45 degrees)

Time integration is projected (forward) Euler: take a tangential step, then
renormalize each token onto the sphere.

## Layout

```
export/
├── tfdy/            small support package: the flow integrator (optim_diracs.usa_flow) and 3D sphere plotting
├── figures/         one script per paper figure (outputs go to results/figures/)
├── verification/    scripts that check specific theorems/claims numerically (print to stdout)
└── results/
    ├── figures/     the regenerated paper figures
    └── logs/        stdout of every script from our reference run
```

`results/` is generated output. You can delete it; the scripts recreate
`results/figures/` when they run.

## Setup

```bash
cd export
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14, numpy 2.4, scipy 1.17, matplotlib 3.10, torch 2.11, scikit-learn 1.8.
`verification/qk_asymmetry_check.py` additionally needs `transformers` (tested with 5.5)
and downloads GPT-2, BERT, ModernBERT and SmolLM2 weights (licenses: MIT, Apache 2.0,
Apache 2.0, Apache 2.0); it is the only script that uses the network.
Everything runs on CPU. LaTeX is not needed, because the figures use matplotlib mathtext.
`scienceplots` is optional: without it you get a harmless `Scienplots not available!`
warning, and the reference figures were produced without it.

**Run every script from inside `export/` with `PYTHONPATH=.`** so the scripts can import `tfdy`:

```bash
PYTHONPATH=. python figures/limit_cycle_figures.py
```

To run everything and save logs:

```bash
for f in figures/*.py verification/*.py; do
  PYTHONPATH=. python "$f" > "results/logs/$(basename "$f" .py).log" 2>&1
done
```

## Figures

| Paper figure | Script | Output | Key settings | Runtime* |
|---|---|---|---|---|
| Fig. "limit cycles" (a) symmetric, (b) sparse, (c) skewed | `figures/limit_cycle_figures.py` | `symmetric.png`, `sparse.jpg`, `skewed.jpg` | N=400, tau=0.1, 4000 steps; eps=0.1 (sparse), eps=0.04 (skewed); **torch seed 42** | 15 s |
| Fig. "persistence" (mobility-weighted kinetic energy and rotation angle vs. time) | `figures/plot_persistence.py` | `persistence.png` | N=400, sparse A, eps in {0, 0.05, 0.1}, tau=0.05, 6000 steps; **torch seed 42** | 35 s |
| Fig. "heightDecay" (appendix, both panels) and Fig. "ripple" (main text, panel (b) only) | `figures/plot_height_decay.py` | `heightDecay.png`, `heightDecay_b.png` | eps=0, beta=1, N in {48, 192, 768}; tau=0.02, 600 steps (a) and 30000 steps (b); deterministic uniform-ring start | 1 min |
| Fig. "buckle" (m=2 out-of-plane growth vs. eps) | `figures/plot_buckle.py` | `buckleThreshold.png` | N=1000, seed amplitude 5e-3, tau=0.05, 5000 steps (T=250); deterministic | 5 min |
| Fig. "triangle" (ring coarsens to a rotating triangle) | `figures/triangle_figure.py` | `triangle.png` | N=60, eps=0.6, sparse A, tau=0.05; ring panel at 800 steps, triangle at 6000; **torch seed 0** | 10 s |
| Fig. "drift" (true exponential vs. first-order Taylor kernel) | `figures/drift_graph.py` | `driftGraph.jpg` | N=400, tau=0.1, 4000 steps to reach the eps=0 ring; **torch seed 42** | 10 s |
| Fig. "isotropic" (App. "A gradient-flow baseline": energy, speed and rotation for D_s = I) | `figures/plot_isotropic.py` | `isotropic.png` | N=300, D_s=I (beta=1), sparse A, eps in {0, 0.05, 0.1}, tau=0.05, 4000 steps; **numpy seed 0** | 45 s |
| *Candidate appendix figure* (not yet in the paper): measured growth rates of the rotating ring vs the closed forms | `verification/growth_rate_check.py` | `growthRates.png` | eps in {0.1, 0.2, 0.3, 0.5}, N in {200, 1000}, tau=0.05, seed amplitude 1e-4; deterministic | ~10 min (8 cores) |

\*Runtime on a laptop CPU (Apple silicon).

**Seeds.** The five scripts that start from a random cloud fix a seed (`SEED` near the
top of each file: a global torch seed, or a numpy generator in `plot_isotropic.py`).
With the seeds as given, and the package versions above, the regenerated figures match
the ones in the paper. Other seeds change the individual token positions but not the
qualitative picture (ring, rotation, tilt, three clusters). `plot_height_decay.py` and
`plot_buckle.py` start from exact uniform rings, so they involve no randomness.

**Figures regenerated for the revised paper.** `persistence.png` (panel (a) relabelled
"mobility-weighted kinetic energy"; `A_sparse` now has the paper's sign, so the rotation
in panel (b) is positive) and `buckleThreshold.png` (panel titles only; same curves)
were regenerated, and `plot_height_decay.py` now also writes `heightDecay_b.png`.

Note on `plot_buckle.py`: the m=2 mode grows at every eps ≠ 0, at rate
λ2 = I2(κ)/I0(κ) ≈ κ²/8. The dashed lines in panel (b) mark where it first doubles
within the window T=250; these apparent thresholds (eps_c ≈ 0.15 aligned, ≈ 0.05
skewed) are an artefact of the window (Remark "finite-time artefact").

## Verification scripts

These scripts print their results. Each one checks a specific statement in the paper.
Scripts that use random numbers seed numpy/torch internally, so their output is
reproducible.

| Script | What it checks (paper reference) | Randomness / knobs | Runtime |
|---|---|---|---|
| `verify_V2.py` | Closed forms of V1 and V2 match the exact field, via an even/odd finite difference in eps. The error scales as eps^2. (Theorem V2, App. "second-order velocity field") | `np.random.seed(0)` | <1 s |
| `V2_inplane.py` | The ring's drift speed is odd in eps (no eps^2 correction), and V2 only modulates density in the plane. (Remark "Second order") | deterministic | <1 s |
| `circulation_check.py` | The non-gradient obstruction extends from D_s=0 to D_s=diag(0,0,1): the circulation of V1 around the equator is (2π/n)(ω·e3) for both, i.e. +2π/3 for A_sparse and +2π for A_skewed. (Thm. "V1 is non-gradient" and the text after it) | deterministic (Fibonacci sphere) | 3 s |
| `planar_invariance_check.py` | Part 1: great circles are exactly invariant (out-of-plane velocity ~1e-16) for random eps, beta, N, omega. Part 2: settled states from random starts (aligned: no tilt; skewed at eps=0.7: three clusters, outside the ring regime). Part 3: an exactly planar tilted ring keeps its tilt until it buckles. (Prop. "planar invariance", Cor. "geometry") | rng seed 0; settle seeds 0–4 | 45 s |
| `tilt_scaling.py` | The skewed ring's tilt from e3 grows approximately linearly in eps (tilt/eps between 108 and 134 for eps ≤ 0.04). (Cor. "geometry", proof of Prop. "skew") | seeds 0–2 | 80 s |
| `tilt_check.py` | Tilt direction and size of the skewed ring: the normal moves toward ω×e3 in all 12 runs; at eps=0.04 the tilt is 3.9–5.0° from a random start and 0.90–0.96° from a near-ring start; at eps=0.1, 8.3–10.5° vs 1.74–1.84°; height correlation 0.82–0.91 from random starts at eps=0.1. (Cor. "geometry", Experiments section, proof of Prop. "skew") | seeds 0–2, random and near-ring starts | 3–8 min |
| `ring_linearization.py` | Spectrum of the symmetric flow at the uniform equatorial ring: rates 1, 1/2, 1/2, and the remaining 2N−3 modes are neutral. (Lemma "spectrum") | deterministic; env `PROBE_N` (48), `PROBE_BETA` (1.0) | 2 s |
| `lyapunov_test.py` | L = <(e3·x)^2> is non-increasing at eps = 0, and for eps > 0 up to slow O(eps^2) ripple growth (the buckle); the O(eps) rotation leaves dL/dt unchanged to O(eps^2); near the ring dL/dt = 2λ2 L − 2βL^2 (T3, predicted vs observed agree to within 2%). Note: BETA multiplies the whole matrix here, so κ = BETA·eps. (Lemma "Ldot", Prop. "aligned", Remark "L-scope") | torch seeds 0–5 and 1; env `PROBE_N` (60), `PROBE_BETA` (2.0) | 65 s |
| `ripple_buckle_check.py` | In the paper's convention (κ = eps): dL/dt of an m=2 ripple matches 2λ2 L − 2βL^2 at every amplitude, and small and large ripples both approach L ≈ λ2/β before creeping slowly upward. (Remark "L-scope", Cor. "no-threshold") | deterministic | 10 s |
| `growth_rate_check.py` | Measured growth rates of the rotating ring match the closed forms. Out-of-plane m=2: λ2 = I2(κ)/I0(κ) (Cor. "no-threshold", Prop. "oop-spectrum"); measured/predicted = 0.95 at dt=0.05 for every κ and N, a first-order time-step error: halving dt halves it, and extrapolating to dt→0 recovers the formula to 0.04%. In-plane m=3: μ3 (Prop. "inplane-spectrum") matches to 0.1%, independent of dt and N; it is the fastest mode (Cor. "m3-fastest"). In-plane m=2, 4: Re μ = 0 exactly; the tiny measured slopes equal the forward-Euler artefact (Im μ)² dt/2. The skewed matrix at eps=0.1 gives the same rates as the sparse one at eps=0.3 (same κ) to 0.1%. Writes `growthRates.png`. | deterministic | ~10 min (8 cores) |
| `ring_normal_force_check.py` | On the flat equatorial ring V1 has no e3 component (exactly 0, both closed form and true-field finite difference) for N in {48, 200, 1000}, beta in {0.5, 2}, random ω; the drift is c = ω3/2. (Prop. "drift", Prop. "aligned") | rng seed 0 | <5 s |
| `verify_ring_spectrum.py` | Closed-form growth rates of the rotating ring vs. a brute-force Jacobian: only m=3 is unstable, mu_3/kappa^2 → 3/16, rates independent of beta and N, mu_3/lambda_2 → 3/2, and the nonlinear flow grows at the predicted rate. (App. "Spectrum of the rotating ring", Cor. "m=3 fastest") | deterministic | 3 s |
| `verify_kgon_equilibria.py` | Every regular k-gon on the equator is an exact rotating relative equilibrium (to machine precision), with Omega_2 = 0. Near-k-gon and random starts all end at k=3. (Prop. "kgon", Observation "triangle selection") | rng seed 0; random starts use seeds 100–109 | 60 s |
| `kgon_rate_check.py` | The rotation rate of the k-gon matches the closed form Ω_k = −Σ_r e^{−eps sin θ_r} sin θ_r / Σ_r e^{−eps sin θ_r} exactly, is independent of beta, is 0 for k=2 and ≈ eps/2 for k ≥ 3. (Prop. "kgon") | deterministic | <1 s |
| `triangle_selection_sweep.py` | Random starts settle into three equal rotating clusters in all 24 runs: eps in {0.1, 0.3, 0.6, 1.0} × 4 seeds at N=60, plus N in {30, 90} and beta in {0.5, 2}. (Observation "triangle selection", Remark "global status") | seeds 0–3 (eps grid), 0–1 (robustness grid) | 80 s |
| `qk_asymmetry_check.py` | Trained attention is far from symmetric. For every head of four models, r = ‖antisym‖_F/‖sym‖_F of the d×d form M_h = W_Q,h W_K,hᵀ/√d_head (d = 768, or 960 for SmolLM2; grouped-query heads paired with their shared key head; rotary embeddings ignored, i.e. same-position score). Median r: GPT-2 0.962, BERT 0.870, ModernBERT 0.922, SmolLM2 0.979; random matrices of the same shape give 0.999. Decoders are slightly less symmetric than encoders of the same era. (Intro, App. "Asymmetry of trained query–key matrices") | none (pretrained weights; random baseline uses numpy seed 0). Needs `transformers`; downloads ~2.3 GB on first run, or set `QK_MODEL_DIR` to local copies | 20 s after download |
| `value_matrix_compare.py` | V = −Id (this paper) vs V = −D (Burger et al.'s choice) in the ring setup: V = −Id spins at eps/2 on an evenly filled ring; V = −D spins about 100× slower, in the opposite sense, and the ring is unevenly filled. (Remark "Effect of the value matrix") | numpy seed 0 | 40 s |

Scripts with `PROBE_N` / `PROBE_BETA` can be rerun at other sizes, e.g.
`PROBE_N=96 PYTHONPATH=. python verification/ring_linearization.py`.

## Conventions

- **Sign of `A_sparse`.** The paper's `A_sparse = [[0,-1,0],[1,0,0],[0,0,0]]` has dual
  axis +e3. `plot_persistence.py`, `circulation_check.py`, `plot_isotropic.py`,
  `kgon_rate_check.py`, `value_matrix_compare.py`, `ripple_buckle_check.py` and the
  scripts that build A with `A_from_axis` use exactly this matrix. Eight older scripts
  (`limit_cycle_figures.py`, `triangle_figure.py`, `drift_graph.py`, `plot_buckle.py`,
  `lyapunov_test.py`, `planar_invariance_check.py`, `V2_inplane.py`, `verify_V2.py`)
  use its transpose, with dual axis −e3; this is marked in each file. It only reverses
  the sense of rotation; every magnitude they report (tilts, growth rates, drift
  speeds, heights) is unaffected, and the figures they produce are the ones in the paper.
- **Where beta sits.** The paper puts beta inside D_s = diag(0, 0, beta), so the coupling
  is κ = eps (ω·e3). `lyapunov_test.py` multiplies the whole matrix by `BETA`, so there
  κ = BETA·eps.

## The `tfdy` package

`tfdy` is adapted from the MIT-licensed code accompanying Burger, Kabri, Korolev,
Roith and Weigand, *Analysis of mean-field models arising from self-attention dynamics
in transformer architectures with layer normalization*. We only kept the modules the
figure scripts use. The main change is `tfdy/optim_diracs.py::usa_flow`, which now
takes

- `A`, `epsilon`: the antisymmetric perturbation, giving `D + epsilon * A`;
- `value`: the value matrix (the paper uses the identity, together with `sgn=+1`,
  i.e. V = −Id);
- `normalize`: softmax-normalized attention (`True`, the paper's model) vs. the
  unnormalized kernel of the original code.

The verification scripts and `plot_buckle.py` / `plot_height_decay.py` /
`plot_persistence.py` do not use `tfdy`. They re-implement the same
flow in a few lines of numpy/torch, so each one can be read on its own.
