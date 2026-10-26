# Data

Every table here is committed and is the only input of the classification, the figures and the generated documents. Raw trajectories are not in the repository ([docs/data.md](../docs/data.md)). Floats are written with full precision (`repr`); booleans are `true`/`false`; empty cells are missing values. Units: D in cm<sup>2</sup> s<sup>−1</sup>, MSD in Å<sup>2</sup>, σ in mS cm<sup>−1</sup>, temperatures in K, times in ps.

| File | Rows | Content |
|---|---:|---|
| `compositions.csv` | 62 | Project 1 identity and screening fields, simulated cell, structure checksum, census evidence labels |
| `structures/` | 62 | MD input supercells (CIF) |
| `runs.csv` | 338 | one row per analysis record (186 census, 54 replicate re-analyses, 98 definitive) |
| `msd_curves.csv.gz` | 24,732 | Li (x, y, z, total) and framework MSD against lag for every record |
| `window_sensitivity.csv` | 1,792 | D in alternative time windows and covariance limits |
| `replicates/` | 6 / 18 / 21 | decisions, per-seed fits and per-temperature summaries of the six census leads |
| `classification.csv` | 62 | the class of every composition (written by `python -m ssemd classify`) |
| `classification_summary.json` | | class counts, prefactor and failed-fit audits, input checksums |
| `workflow.csv` | 13 | counts from 21,712 Materials Project entries to the classes |
| `references/li6ps5cl_calibration.csv` | 3 | σ(300 K) of Li<sub>6</sub>PS<sub>5</sub>Cl from the separate MLIP benchmark |
| `frozen/` | 3 | the published classification, its summary and run records, byte for byte as produced (with `SHA256SUMS`) |
| `provenance/` | | trajectory checksums, analysis-settings hashes and the map from every file to its original source |

## runs.csv

| Column | Meaning |
|---|---|
| `record_id` | `<stage or batch>/<candidate_key>/<T>K_s<seed>` |
| `trajectory_id` | trajectory analysed (`<md_batch>/...`); the 18 seed-0 trajectories of the six leads have a census record and a replicate record |
| `stage` | census, replicate, definitive |
| `md_batch` | census_12, census_50, replicate_seed12, def_b1, def_b2, def_b3 |
| `estimator` | kinisi_mcmc or fixed_cov_analytic ([methods](../docs/methods.md#diffusion-coefficients)) |
| `protocol_section` | census, replicate validation, compute list + A1, A2, A3 |
| `production_ps` | production length after 10 ps equilibration |
| `n_atoms`, `n_Li`, `cell_volume_A3` | simulated cell |
| `temperature_mean_K`, `temperature_bias_fraction` | thermostat check |
| `D_cm2_s`, `D_lo_cm2_s`, `D_hi_cm2_s` | estimator D and 95 % interval |
| `D_direct_cm2_s` | direct MSD slope over 60–100 ps |
| `Li_msd_100ps_A2`, `D_bound_cm2_s` | Li MSD at 100 ps and the bound MSD/(6 × 100 ps) |
| `framework_msd_100ps_A2`, `framework_fraction_gt2A_200ps` | framework integrity |
| `diffusion_resolved`, `framework_intact` | census criteria |
| `eligible_census_criteria` | resolved, intact and \|T bias\| ≤ 5 % |
| `kinisi_fit_failed`, `eligible` | amendment A3.2 and the final eligibility |
| `ne_factor_mS_cm_per_cm2_s` | σ = factor × D at the run temperature |
| `energy_slope_meV_atom_ps`, `D_pymatgen_cm2_s` | census only |
| `review_flags` | census flags |
| `used_by_classifier`, `superseded_by` | the 318 records the classifier reads; others point to the record that replaced them |
| `trajectory_sha256` | sha256 of the trajectory file |
| `legacy_sources` | original tables of the record (reproduces the frozen `data_sources` column) |

## classification.csv

The classifier columns of the protocol (class, eligible and tested temperatures, E<sub>a</sub> and its kind, D<sub>0</sub>, σ(300 K) and its kind, pooled 95 % interval, prefactor sensitivity, A3.2 audit, basis text, flags) plus `decided_by`, `sigma300_estimator` and `qualifiers` ([methods](../docs/methods.md#vocabulary)).
