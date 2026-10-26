# Methods

## Input structures

The 62 compositions are the frozen candidate queue of Project 1 ([Solid-Electrolyte-Screening](../README.md#related-projects)): all 21,712 Li-containing Materials Project entries (database 2026.04.13) were filtered to 248 near-stable, insulating Li–S entries (S present, no H, 3–5 elements, E<sub>hull</sub> ≤ 0.05 eV/atom, band gap ≥ 1.5 eV), ranked with a conductivity model trained on 599 experimental records, and grouped into 62 compositions. One representative Materials Project structure per composition was simulated at its MP/PBE volume (no relaxation). Supercells are diagonal integer replications with every perpendicular cell height ≥ 18 Å (216–1000 atoms). Identity, cell and checksum of every input: [data/compositions.csv](../data/compositions.csv) and [data/structures/](../data/structures/).

## Molecular dynamics

- Potential: MACE-MP-0 small, not fine-tuned, float32 (model sha256 `2ddb079c…`).
- NVT Langevin dynamics (friction 0.01 in ASE units), 1 fs time step, 10 ps equilibration, one frame every 50 fs.
- Initial velocities and Langevin noise are seeded with the MD seed (`np.random.seed(md_seed)`), so replicates differ only by the seed.
- Code: [production/md_runner/md.py](../production/md_runner/md.py). All runs used mace-torch 0.3.16, torch 2.5.0 and ase 3.29.0 in one container on NVIDIA A100-40GB GPUs of NSCC ASPIRE 2A.

| Stage | Trajectories | Temperatures (K) | Production | MD seeds |
|---|---:|---|---|---|
| Census | 186 | 600, 800, 1000 | 200 ps | 0 |
| Replicates of the six census leads | 36 | 600, 800, 1000 | 200 ps | 1, 2 |
| Definitive batch 1 (compute list + A1) | 29 | 400–700, 600–1000 for two leads | 200–800 ps | 0, 3, 4 |
| Definitive batch 2 (A2) | 45 | 400–800 | 200–800 ps | 0, 1, 2 |
| Definitive batch 3 (A3) | 24 | 400–800 | 200–800 ps | 3, 4 |

Every task is listed in [production/manifests/](../production/manifests/) with the sha256 of its trajectory.

## Diffusion coefficients

**kinisi estimator (census and definitive batches).** Positions are unwrapped with minimum-image frame differences and the mean displacement of the non-Li framework is removed at every frame. The Li mean-squared displacement is fitted with the kinisi Bayesian estimator: 60 lags, fit start at 30 % and maximum lag at 50 % of the window, covariance condition limit 10<sup>8</sup>, 32 emcee walkers × 1000 steps after 500 burn-in, thinning 10 (3200 retained draws). D is the posterior median and the interval the equal-tail 95 % credible interval, conditional on the estimator. The full trajectory is the primary window; halves, three quarters and (for runs longer than 200 ps) the first 200 ps are sensitivity windows, and the census also compares condition limits 10<sup>6</sup> and 10<sup>10</sup> ([data/window_sensitivity.csv](../data/window_sensitivity.csv)). Code: [src/ssemd/transport/kinisi_census.py](../src/ssemd/transport/kinisi_census.py); its settings hash equals the one recorded by every definitive analysis.

**Fixed-covariance estimator (replicates of the six leads).** All seeds of the six leads, including their seed-0 census trajectories, were re-analysed with one estimator: time origins every 1 ps over the full 200 ps, lags 60–100 ps on a 1 ps grid, the kinisi covariance matrix at condition limit 10<sup>8</sup>, and an analytic generalised-least-squares posterior with a free intercept and a non-negative slope ([src/ssemd/transport/fixed_cov.py](../src/ssemd/transport/fixed_cov.py)). On the same 18 seed-0 trajectories the two estimators agree closely (Fig. 5e).

**Direct-MSD cross-check.** A linear fit to the Li MSD over 60–100 ps lags, D = slope/6.

**Conductivity.** Nernst–Einstein, σ = n<sub>Li</sub> e<sup>2</sup> D / (V k<sub>B</sub> T), with the Li count and volume of the simulated cell; ion correlations are neglected.

## Eligibility of a run

A run is eligible when (census criteria, unchanged):

- Li diffusion is resolved: positive lower credible D, Li MSD growth ≥ 1 Å<sup>2</sup> over the fit region and a log–log MSD slope of 0.5–1.5;
- the framework is intact: non-Li MSD at 100 ps lag ≤ 1 Å<sup>2</sup> and at most 10 % of non-Li atoms displaced by more than 2 Å at 200 ps;
- the mean temperature is within 5 % of the target;
- (amendment A3.2) the kinisi D agrees with the direct-MSD D within a factor 3; otherwise the kinisi fit is treated as failed and the run as not resolved.

A temperature is eligible when at least one of its runs is eligible. Melted or unresolved temperatures are excluded from fits but never veto a composition.

## Classification

The rules are fixed in the pre-registered [classification protocol](CLASSIFICATION_PROTOCOL.md) and implemented in [src/ssemd/rules.py](../src/ssemd/rules.py):

- **σ(300 K)**: OLS fit of ln D against 1/T over all eligible runs, extrapolated to 300 K (Nernst–Einstein at 300 K); with ≥ 3 points the 95 % confidence interval of the mean line at 300 K is reported. Fast threshold: 0.1 mS/cm.
- **Replicate rule**: ≥ 3 usable seeds at every fitted temperature, between-seed D max/min ≤ 2, and per-seed and pooled Arrhenius R<sup>2</sup> ≥ 0.95. Once replicate seeds exist the rule is applied before any slow class (amendment A3.1). For the six census leads the replicate decision of the seed 0/1/2 analysis is used (σ and E<sub>a</sub> are means of per-seed fits; Li4Re6S11 on seeds 0 and 1, with seed 2 excluded because its 800 K fit depended on the fit region).
- **Terminal rule**: if the replicate rule fails, seeds are added up to five or runs extended to 800 ps; then the pooled 95 % interval decides (lower limit ≥ 0.1: fast; upper limit < 0.1: slow; otherwise the pooled value decides and the class is flagged `near_threshold`).
- **Bounds**: a non-resolved, framework-intact run gives D ≤ MSD<sub>Li</sub>(100 ps)/(6 × 100 ps). With an assumed prefactor D<sub>0</sub> ≥ 10<sup>−4</sup> cm<sup>2</sup> s<sup>−1</sup> this bounds E<sub>a</sub> from below and σ(300 K) from above; one eligible temperature with a lower bounded temperature also gives a two-point slope bound that does not assume D<sub>0</sub>. Every bounded class is re-evaluated at D<sub>0</sub> ≥ 10<sup>−5</sup> cm<sup>2</sup> s<sup>−1</sup>.

| Class | Rule |
|---|---|
| FAST_VALIDATED | replicate rule (or terminal rule) and σ(300 K) ≥ 0.1 mS/cm |
| SLOW_MEASURED | fit over ≥ 3 eligible temperatures, σ(300 K) < 0.1 mS/cm |
| SLOW_ESTIMATED | exactly 2 eligible temperatures, two-point σ(300 K) < 0.1 mS/cm |
| SLOW_BOUNDED | fewer than 2 eligible temperatures, upper bound < 0.1 mS/cm |
| THERMALLY_UNSTABLE | no intact framework at any tested temperature down to 400 K |

## Vocabulary

The class is the only verdict. Earlier labels are kept as evidence columns and are never used as results:

| Column | Values | Meaning |
|---|---|---|
| `census_screen_label` | conditional, review, not_reliable | seed-0 census label; one failed temperature made a composition `not_reliable` |
| `census_category` | lead, unresolved_diffusion, framework_melting, both, undetermined_600K | census reason for compositions that were not leads |
| `replicate_outcome` | pass, pass_seeds01, pass_low_sigma, method_pending | seed 0/1/2 decision of the six census leads |
| `decided_by` | six_lead_validation, replicate_rule, terminal_rule, two_point_fit, prefactor_bound, pair_bound | rule that placed the composition |
| `qualifiers` | near_threshold, a32_dependent, d0_dependent, upper_value_estimate | caveats on a class |

## What the numbers mean

- All values are conditional on one un-fine-tuned universal potential at the MP/PBE volume, one representative structure per composition and one supercell size. On Li<sub>6</sub>PS<sub>5</sub>Cl the same MACE-MP-0 protocol gives σ(300 K) = 5.55 mS/cm against 3.15 mS/cm measured (about 1.8×), MatterSim gives 0.35 and a DFT-fine-tuned MACE 0.45 mS/cm ([data/references/li6ps5cl_calibration.csv](../data/references/li6ps5cl_calibration.csv); separate benchmark project). Activation energies and the ordering are more reliable than absolute σ.
- Universal MLIPs soften the potential-energy surface (Deng et al., [arXiv:2405.07105](https://arxiv.org/abs/2405.07105)), which can make frameworks melt in simulation at temperatures where the material is stable; excluding melted temperatures follows high-throughput practice (Wang et al., *Nat. Commun.* 2023, [doi:10.1038/s41467-023-43436-3](https://doi.org/10.1038/s41467-023-43436-3)).
- Posterior draws, time windows and MSD origins are not independent MD replicates; intervals do not include force-field, finite-size or structural uncertainty.
- Amendment A3 was made after the batch-2 results were seen (and before batch 3 ran); two fast classes depend on its failed-fit rule (A3.2) and are marked `a32_dependent`.
