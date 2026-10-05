# Code for "From Rest to Rotation: Antisymmetric Attention Creates Rotating Steady States in Mean-Field Transformer Dynamics"

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
├── run_review_experiments.sh   runs the experiments E1–E11 below in sequence, with logs
├── check_progress.sh           shows which of them is running
└── results/
    ├── figures/     the regenerated paper figures
    ├── logs/        stdout of every script from our reference run
    └── review/      JSON results and logs of the experiments E1–E11
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
Four scripts read pretrained weights from the Hugging Face hub (licenses: GPT-2 MIT; BERT,
ModernBERT, SmolLM2 Apache 2.0) and are the only ones that use the network:
`qk_asymmetry_check.py` and `ov_sign_check.py` (all four models), and
`trained_head_flow.py` and `antisym_depth.py` (GPT-2 and BERT; these also run the models
on two short text passages, so they need `transformers`; tested with torch 2.4 and
transformers 4.41). Once the weights are cached, set `HF_HUB_OFFLINE=1`. Behind a proxy that
inspects HTTPS, point `REQUESTS_CA_BUNDLE` at your system certificates (the runner does this
on macOS).
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
| `isotropic_rotation_check.py` | The isotropic minimizer rotates (Prop. "iso-rot"): for D = beta·I + eps·A the field at the uniform measure is exactly eps·ρ(r)/r·Ax, ρ(r) = coth r − 1/r, a pure rotation about ω. Part 1: direct quadrature agrees to ≤ 4e-5 relative error (pure quadrature error: it falls 4× per grid doubling). Part 2: the predicted spin rates 0.01565 and 0.03129 equal those measured by `plot_isotropic.py`. | rng seed 1 (test points) | 6 s |
| `cluster_stability_check.py` | Linear stability of the rotating k-cluster states (Prop. "kstab"): the finite-difference Jacobian matches the closed-form spectrum to ≤ 2e-8; growing eigenvalues k=2: 0, k=3: 0, k=4: 1, k=5: 2N−6, k=6: 3, so only k=2, 3 are linearly stable (k=2 static, k=3 rotating); for k ≥ 4 the growth rate is eps²/8 (eps²/4 for k=4); a tilted triangle keeps its tilt while its clusters tighten. | numpy seed 0 (Part 4 noise) | 5 s |
| `ring_normal_force_check.py` | On the flat equatorial ring V1 has no e3 component (exactly 0, both closed form and true-field finite difference) for N in {48, 200, 1000}, beta in {0.5, 2}, random ω; the drift is c = ω3/2. (Prop. "drift", Prop. "aligned") | rng seed 0 | <5 s |
| `verify_ring_spectrum.py` | Closed-form growth rates of the rotating ring vs. a brute-force Jacobian: only m=3 is unstable, mu_3/kappa^2 → 3/16, rates independent of beta and N, mu_3/lambda_2 → 3/2, and the nonlinear flow grows at the predicted rate. (App. "Spectrum of the rotating ring", Cor. "m=3 fastest") | deterministic | 3 s |
| `verify_kgon_equilibria.py` | Every regular k-gon on the equator is an exact rotating relative equilibrium (to machine precision), with Omega_2 = 0. Near-k-gon and random starts all end at k=3. (Prop. "kgon", Observation "triangle selection") | rng seed 0; random starts use seeds 100–109 | 60 s |
| `kgon_rate_check.py` | The rotation rate of the k-gon matches the closed form Ω_k = −Σ_r e^{−eps sin θ_r} sin θ_r / Σ_r e^{−eps sin θ_r} exactly, is independent of beta, is 0 for k=2 and ≈ eps/2 for k ≥ 3. (Prop. "kgon") | deterministic | <1 s |
| `triangle_selection_sweep.py` | Random starts settle into three equal rotating clusters in all 24 runs: eps in {0.1, 0.3, 0.6, 1.0} × 4 seeds at N=60, plus N in {30, 90} and beta in {0.5, 2}. (Observation "triangle selection", Remark "global status") | seeds 0–3 (eps grid), 0–1 (robustness grid) | 80 s |
| `qk_asymmetry_check.py` | Trained attention is far from symmetric. For every head of four models, r = ‖antisym‖_F/‖sym‖_F of the d×d form M_h = W_Q,h W_K,hᵀ/√d_head (d = 768, or 960 for SmolLM2; grouped-query heads paired with their shared key head; rotary embeddings ignored, i.e. same-position score). Median r: GPT-2 0.962, BERT 0.870, ModernBERT 0.922, SmolLM2 0.979; random matrices of the same shape give 0.999. Decoders are slightly less symmetric than encoders of the same era. (Intro, App. "Asymmetry of trained query–key matrices") | none (pretrained weights; random baseline uses numpy seed 0). Needs `transformers`; downloads ~2.3 GB on first run, or set `QK_MODEL_DIR` to local copies | 20 s after download |
| `value_matrix_compare.py` | V = −Id (this paper) vs V = −D (Burger et al.'s choice) in the ring setup: V = −Id spins at eps/2 on an evenly filled ring; V = −D spins about 100× slower, in the opposite sense, and the ring is unevenly filled. (Remark "Effect of the value matrix") | numpy seed 0 | 40 s |

Scripts with `PROBE_N` / `PROBE_BETA` can be rerun at other sizes, e.g.
`PROBE_N=96 PYTHONPATH=. python verification/ring_linearization.py`.

## Experiments E1–E11 (value matrix, trained heads, long runs)

These scripts write JSON to `results/review/` and a log to `results/review/logs/`.
`zsh run_review_experiments.sh` runs all of them in sequence at low priority
(`WORKERS` sets the number of processes, default 2; `PY` is the python for the numpy-only
scripts, `PYHF` the one with torch + transformers). `zsh check_progress.sh` shows progress.
All are deterministic given the seeds in each file.

| ID | Script | What it checks (paper reference) | Output | Runtime* |
|---|---|---|---|---|
| E1 | `value_matrix_sweep.py` | Long-time state under V = −Id, +Id, −D, +D, for D_s = diag(0,0,1) and D_s = Id, from random and ring starts: only V = −Id keeps moving. (Sec. "role of the value matrix", Table "value-sims") | `E1_value_matrices.json` | 15 min |
| E2 | `ov_sign_check.py` | Sign of the value–output map of every head of GPT-2, BERT, ModernBERT, SmolLM2: 33–43% push on balance. (App. "Sign of the value–output map", Table "ov") | `E2_ov_sign.json` | 1 min after download |
| E3 | `triangle_sweep_large.py` | 780 runs over eps, N (incl. N not divisible by 3), beta and start type: all end in three dominant clusters. (Obs. "triangle selection", App. "triangle sweep") | `E3_triangle_sweep.json` | 1–2 h |
| E3b | `triangle_longtime.py` | 80 long runs (N = 60, 61), with and without 1e-8 kicks: three groups remain; at N = 61, eps = 0.1 they break up and re-form. (App. "triangle sweep", Remark "unequal triangles") | `E3b_triangle_longtime.json` | 30 min |
| E4 | `integrator_check.py` | RK4 vs forward Euler: growth rates to 0.04% with RK4; Euler forms the triangle about 13% earlier but reaches the same state. (App. "growth", "settings") | `E4_integrator.json` | 5 min |
| E5 | `highdim_check.py` | The ring and triangle on S^4 spin at the predicted rates; with a 3-sphere resting set the tokens still end on one great circle. Writes `results/figures/highdim.png`. (App. "higher dimensions") | `E5_highdim.json` | 30 min |
| E6 | `lemma_fix_check.py` | The ripple law of Lemma "Ldot" with its G term, the general near-ring law, the polar lemma, and Prop. "VD" (zero circulation for V = −D). | `E6_lemma_checks.json` | <1 min |
| E7 | `trained_head_flow.py` | All 288 heads of GPT-2 and BERT run as the flow with their own weights, real token states, D vs its symmetric part, and V, its symmetric part, or its pushing part, to T = 1000: heads with a pulling direction collapse; the 12 pure-push heads keep moving, ~9× faster with the antisymmetric part. Summary: `SUMMARY=results/review/E7_trained_heads.json python verification/trained_head_flow.py`. (App. "trained heads", Table "trained") | `E7_trained_heads.json` (`E7_gpt2.json`: an independent GPT-2 rerun, identical) | 3–5 h (4 workers) |
| E8 | `lowrank_value_check.py` | V = diag(−1,−1,c): the ring rotates at the same rate for every c, is held for c < 0, neutral at c = 0, and collapses for c > 0; S^4 with rank-2 push. `EXT=1` adds c = ±0.1 and long runs. (App. "partial and mixed value matrices") | `E8_lowrank_value.json`, `E8b_lowrank_ext.json` | 10 + 20 min |
| E9 | `unequal_triangle_check.py` | Unequal rotating triangles (21/20/19, ...) exist; their linear stability depends on eps. (Remark "unequal triangles") | `E9_unequal_triangle.json` | 2 min |
| E10 | `antisym_share.py` | Share of each trained head's token motion removed by symmetrizing D (about half), against a random antisymmetric part of equal norm. Needs the E7 cache. (App. "trained heads") | `E10_antisym_share.json` | 1 min |
| E11 | `antisym_depth.py` | Each trained head iterated for 1–48 layers with its real per-layer step: after 12 layers the antisymmetric part moves a token by ~19° and changes a third of nearest neighbours. Needs the E7 cache. (App. "trained heads") | `E11_antisym_depth.json` | 3 min |

\*Laptop CPU. `trained_head_flow.py` first builds `results/review/E7_heads.npz` (~650 MB: the
head matrices and token states), and `antisym_depth.py` builds `E11_steps.npz`; both caches are
regenerated automatically and are not included in the archive.

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
