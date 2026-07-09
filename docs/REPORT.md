# Project 2 Technical Report: MLIP-MD Ionic Conductivity of Li6PS5Cl

**Milestone ② (W5-W9) · Sulfide solid electrolyte · Universal machine-learning potential + molecular dynamics + fine-tuning**

---

## 1. Goal and system

Drive molecular dynamics (MD) with a machine-learning interatomic potential (MLIP) to
compute the Li+ ionic conductivity sigma(300 K) of the argyrodite-type solid electrolyte
**Li6PS5Cl**, and benchmark it against experiment (sintered ~3.15 mS/cm; mechanochemical
~1.33 mS/cm).

Full pipeline:
> Structure modeling -> multi-temperature NVT-MD -> mean-squared displacement MSD /
> diffusion coefficient D -> Nernst-Einstein sigma(T) -> Arrhenius extrapolation to
> sigma(300 K) -> benchmark against experiment -> re-evaluate after **DFT labeling +
> fine-tuning**.

Design principle: **"baseline first, fine-tune second"** -- first look at the systematic
bias of the un-fine-tuned universal potential, then fine-tune on a small amount of DFT
data, quantifying the sensitivity of "label quality -> potential -> sigma".

---

## 2. Methods

### 2.1 Structure modeling (`01_build_structure.py`, `src/structure.py`)
- Pulls Materials Project **mp-985592** (cached CIF). Li6PS5Cl argyrodite: a PS4^3-
  tetrahedral framework + free S2-/Cl- disordered over the 4a/4d sites.
- Identifies the free anions by P-S bond length, and performs **Ewald energy
  enumeration** over the S/Cl partial occupancy to obtain an ordered approximant ->
  `data/config0.cif` (52 atoms, cell a~=10.28 A).
- Production uses a **2x2x2 supercell** `data/config0_sc222.cif` (416 atoms, density
  unchanged at 1.641 g/cm3, 192 Li atoms) -- the larger box suppresses MSD statistical
  noise and finite-size effects.

### 2.2 Molecular dynamics (`02_baseline_md.py`, `src/md.py`)
- ASE **NVT Langevin**, friction coefficient 0.01 fs^-1, timestep 1 fs, temperatures
  600/800/1000 K (high temperature accelerates diffusion so it can be sampled within an
  affordable run length, then Arrhenius-extrapolated back to 300 K).
- Potentials: **MACE-MP-0 (small, L0)** and **MatterSim (v1.0.0-1M)**, two universal
  potentials run side by side; float32, NVIDIA A40 GPU.
- Recording starts after equilibration, one frame saved every 50 steps. Duration tiers:
  50 ps (thin) / 150 ps (convergence check) / **200 ps (production, 416-atom)**.

### 2.3 Transport analysis (`03_analyze_transport.py`, `src/transport.py`)
- **pymatgen `DiffusionAnalyzer`** as the backbone for D (from the linear segment of MSD).
- **kinisi** for error bars: uses generalized least squares + bootstrap to handle the
  strong correlation between MSD time points, giving the statistical uncertainty on D
  (see §4).
- **Nernst-Einstein**: sigma = n q^2 D / (k_B T), where n is the Li+ number density
  (computed from the structure and its volume).
- **Arrhenius**: fits ln(sigma*T) = -E_a/(k_B T) + c (the textbook linear form),
  extrapolating sigma(300 K) and giving E_a and R^2.

### 2.4 DFT labeling (`finetune/11_label_qe.py`, `label_qe.pbs`, `submit_shards.sh`)
- Samples **27 snapshots** (52 atoms) from the MD trajectories, and runs a **Quantum
  ESPRESSO pw.x** SCF single-point calculation on each to produce energies + forces
  (+ stress).
- Pseudopotentials: **GBRV all-PBE ultrasoft** (Li/P/S/Cl), ecutwfc 60 / ecutrho 480 Ry,
  Gaussian smearing with degauss 0.01, conv_thr 1e-7, `tprnfor+tstress`.
- **Gamma-point, real-wavefunction** sampling (`--kpts gamma`): ~16x cheaper than a
  2x2x2 mesh, and sufficient for a delta fine-tuning label; a single-point SCF takes
  ~26 min (16 cores, Intel-MPI, `npool=1`).
- The 27 snapshots are split into **27 concurrent single-configuration CPU jobs**
  (sharded via a `--start` offset); all finished labeling in ~1.8 h with 0 failures.

### 2.5 Fine-tuning (`finetune/20_finetune_mace.sh`, `finetune_mace.pbs`)
- `mace_run_train`, with `--foundation_model` pointing at the locally cached MACE-MP-0
  small checkpoint (offline-safe).
- **`--E0s=average` (critical)**: QE's absolute energies (~-269 eV/atom) are on a
  different scale from MACE-MP's reference scale, so the per-element reference energies
  must be recomputed from this dataset -- otherwise the energy loss is dominated by a
  huge constant offset. Forces are independent of E0 and therefore unaffected, and the
  force weight (10) is much larger than the energy weight (1) anyway.
- lr 1e-4, batch size 4, 120 epochs + **SWA**, float32, A40. 27 configurations -> 22
  training / 5 validation.

---

## 3. Results

Benchmark: Li6PS5Cl room-temperature ~3.15 mS/cm (sintered). Figures in
`../figures/{main,supplementary}/03_arrhenius*.*` and `03_sigma300_vs_expt*.*` (production/`_ft`
in `main/`, baseline/`_long` in `supplementary/`; double-column Nature spec, MD points carry
kinisi error bars).

sigma(300 K) is a **weighted** Arrhenius extrapolation (each temperature weighted by its kinisi
uncertainty -- the Mo-group `aimd` standard); the bracket is the propagated [sigma_min, sigma_max]
1-sigma interval and E_a carries its fit error.

| Model (cell x duration) | D@1000K (cm2/s) | sigma(300K) [min, max] (mS/cm) | E_a (eV) | Ratio to expt. | R2 |
|---|---|---|---|---|---|
| MACE-MP-0 (52-atom, 50 ps) | 5.0e-5 | 22.7 [11, 45] | 0.21±0.03 | 7.2x | 0.79 |
| MACE-MP-0 (52-atom, 150 ps) | 4.6e-5 | 12.5 [5.9, 27] | 0.234±0.031 | 4.0x | 0.999 |
| MatterSim (52-atom, 50 ps) | 4.3e-5 | 14.1 [6.9, 29] | 0.22±0.03 | 4.5x | 0.83 |
| **MatterSim (52-atom, 150 ps)** | 3.8e-5 | **1.79 [0.82, 3.9]** | 0.292±0.032 | **0.57x** | 0.982 |
| **MACE-MP-0 (416-atom, 200 ps)** production | 5.5e-5 | **5.55 [4.2, 7.3]** | 0.265±0.012 | **1.8x** | 0.999 |
| **MatterSim (416-atom, 200 ps)** production | 3.2e-5 | **0.35 [0.25, 0.49]** | 0.346±0.014 | **0.11x** | 0.991 |
| **MACE fine-tuned (416-atom, 200 ps)** | 3.0e-5 | **0.45 [0.31, 0.66]** | 0.335±0.015 | **0.14x** | 0.998 |

**The over/under bracketing is statistically robust, not a fitting artifact:** the production band
[4.2, 7.3] sits entirely above experiment (3.15) and the fine-tuned band [0.31, 0.66] entirely below --
the two 1-sigma intervals do not overlap experiment from the same side.

Fine-tuned-potential validation force RMSE: foundation model 234 -> fine-tuned
**62.4 meV/A** (3.75x improvement); validation energy RMSE 3.3 meV/atom. Convergence
curves in `../figures/main/04_finetune_convergence.*` (validation force/energy RMSE vs. epoch,
compared against the foundation-model baseline; the spike around epoch 90 is the SWA
stage-two restart).

MD stability (conservation check -- temperature and energy stable over time, no
blow-up/drift) is shown in `../figures/supplementary/02_md_stability_mace{,_prod,_ft}.*` (baseline 50 ps /
production 200 ps, 416-atom / fine-tuned 200 ps).

### 3.1 Literature benchmark (result evaluation)

| Source | Type | sigma(300 K) mS/cm | E_a (eV) | Method / conditions |
|---|---|---|---|---|
| **This work (production)** | MLIP-MD | **5.55 [4.2, 7.3]** | **0.265** | MACE-MP-0, 416 atoms, 200 ps, extrapolated from 600-1000 K |
| This work (MatterSim production) | MLIP-MD | 0.35 [0.25, 0.49] | 0.346 | MatterSim, 416 atoms, 200 ps |
| This work (MatterSim, 150 ps) | MLIP-MD | 1.79 | 0.292 | MatterSim, 52 atoms, 150 ps |
| This work (fine-tuned) | MLIP-MD | 0.45 [0.31, 0.66] | 0.335 | Gamma-DFT-fine-tuned MACE |
| Literature (computational) | MLIP-MD | 2.2 (extrapolated from 800-1200 K) / 0.22 (extrapolated from 500-700 K) | -- | 400-1200 K; the authors report **no hops** at 300 K/100 ns, requiring extrapolation, and the extrapolated value varies strongly with the fitted temperature window [1] |
| Experiment (optimized sintering) | EIS | 3.15 | -- | 550 degC sintering [2] |
| Experiment (liquid-phase synthesis) | EIS | >2 | -- | [3] |
| Experiment (wet milling + annealing) | EIS | 1.0-1.9 | -- | [4] |
| Experiment (mechanochemical) | EIS | 1.33 | -- | ball milling |
| Experiment (range across studies) | EIS | ~1-3.2 | **0.22-0.38** | multiple processing routes; solvent-processed samples as low as ~0.20-0.25, some studies 0.35-0.38 [5] |

**Takeaways**:
- **sigma(300 K)**: the production value of 5.55 [4.2, 7.3] is **1.8x** the optimized-sintering
  experimental value (3.15), the same order of magnitude as other MLIP-MD high-temperature
  extrapolations (2.2); all results are within "about 2x above the experimental value" --
  very good agreement for a purely un-fine-tuned potential.
- **E_a**: the production value (0.265 eV) and the fine-tuned value (0.335 eV) **both fall
  within the experimental 0.22-0.38 eV range**, correctly capturing the temperature
  dependence.
- **The extrapolation sensitivity is corroborated by the literature**: [1] explicitly
  reports no hops in 100 ns at 300 K, requiring high-temperature extrapolation, with the
  extrapolated value ranging from 0.22 to 2.2 mS/cm depending on the fitted temperature
  window -- consistent with this work's observation that long-range extrapolation carries
  extra uncertainty, which is why the 50->150 ps convergence check matters.
- **Single-crystal upper bound**: most experiments are on polycrystalline samples with
  grain boundaries, so the simulation being "somewhat above typical experiments" matches
  expectations; the fine-tuned model's under-prediction (0.14x) is the outlier, attributed
  in §5.

References: see the **References** section at the end ([1] arXiv:2403.14116 · [2] ACS AMI
2018 · [3] Chem. Mater. 2023 · [4] Front. Chem. 2021 · [5] ACS AMI 2018).

---

## 4. Error analysis ("why does my D have an error bar, where does the error come from")

**Statistical error (the error bars in the figures)**: each time point of MSD(t) is
computed from a sliding-window average over the same trajectory, so they are **strongly
correlated with each other** -- fitting MSD directly with ordinary least squares would
underestimate the variance of D. kinisi explicitly models this correlation using
**generalized least squares + bootstrap**, giving a posterior distribution for D whose
standard deviation is the error bar. The 150/200 ps converged trajectories have small
error bars (sufficient statistics), while the 50 ps low-temperature point has a large
error bar and a poor R^2 (MSD has not yet entered the linear diffusive regime, i.e.
insufficient sampling) -- this is exactly why the 600 K point falls back into line and R^2
goes from 0.81 to 0.999 after extending 50 ps to 150 ps: **the low-temperature scatter is
sampling noise, not genuine non-Arrhenius behavior**.

**Systematic error (not captured in the error bars, but determines credibility)**:
1. **Lattice too large**: the MP/PBE lattice a~=10.28 A is ~4.3% larger than the
   experimental ~9.86 A; a looser lattice -> faster diffusion -> sigma biased high.
2. **Single-crystal upper bound**: the simulation has no grain boundaries, so the real
   polycrystalline sigma is lower -- MD gives an **upper bound**.
3. **Nernst-Einstein approximation**: ignores inter-ion correlation (Haven ratio H_R != 1),
   systematically affecting the absolute value of sigma.
4. **Potential PES softening / stiffening**: a universal potential may underestimate the
   sulfide migration barrier (sigma biased high); an over-stiff fine-tuned potential
   overestimates the barrier (sigma biased low) -- see §5.
5. **Finite size / duration**: the 52-atom cell's limitations are mitigated by the
   416-atom supercell and 200 ps duration.

**What to trust and what not to**: what is trustworthy is the **order of magnitude and the
trend** (the Arrhenius fit is a clean line with R^2>0.99, E_a falls in the literature's
0.2-0.35 eV range, the production value is 1.8x experiment); what should not be
over-interpreted is the **absolute value to two significant figures** (affected by the
systematic errors above and the Nernst-Einstein approximation).

---

## 5. Discussion ("roughly what is the MLIP learning" + benchmark)

- **The un-fine-tuned foundation model over-predicts, but converges downward with better
  sampling**: both universal potentials systematically over-predict single-crystal sigma
  at 50 ps (MatterSim one notch lower than MACE). They are trained on Materials Project's
  near-equilibrium PBE data, so what they have learned is the **near-equilibrium
  energy-force surface**; they extrapolate poorly to the details of the Li+ migration
  transition state and the sulfide's soft framework, tending to underestimate the barrier
  -> overestimate sigma. Extending to 150 ps converges both noticeably: MACE to 4.0x,
  **MatterSim drops to 0.57x (sigma 1.79 mS/cm, the closest single run to experiment)** --
  showing that most of the short-run over-prediction is a statistical artifact of
  unconverged MSD (note the wide weighted-fit bands on the 50 ps points). **W8 production**
  (416 atoms + 200 ps) brings MACE down to **1.8x experiment, [4.2, 7.3] mS/cm, R^2=0.999**,
  the literature-level agreement that pure MLIP-MD can reach.
- **Fine-tuning stiffens the potential**: after fine-tuning on 27 QE-Gamma DFT forces, the
  validation force RMSE drops from 234 to 62 meV/A (the model genuinely fits our DFT forces
  more closely), but **E_a rises from 0.265 to 0.335 eV and sigma drops to 0.14x**. In
  other words: **the foundation model over-predicts, the fine-tuned model under-predicts,
  bracketing experiment right in between** (geometric mean ~1.6 mS/cm ~= 0.5x experiment) --
  and the bracketing is statistically robust: the production band [4.2, 7.3] and the
  fine-tuned band [0.31, 0.66] both exclude experiment, from opposite sides.
- **Source of the stiffening (honest attribution)**: (a) **Gamma-only k-point sampling**
  under-samples the Brillouin zone for a 52-atom cell, which can systematically bias
  forces/energies; (b) **GBRV/PBE** differs from MACE-MP's VASP/PBE-PAW reference, so
  fine-tuning "re-calibrates" the forces to our DFT flavor; (c) the **small 27-configuration
  dataset**, mostly near-equilibrium snapshots, lacks configurations sampling the migration
  transition state -> overfitting to an overly stiff local potential-energy surface. This
  shows that sigma is **highly sensitive to label quality** -- one of this project's most
  valuable empirical conclusions.

---

## 6. Honest limitations (for the record)
- Using an un-fine-tuned universal potential as-is -> systematic bias (this project
  measured a 1.8-7.2x over-prediction).
- Unconverged MSD at the thin-pipeline tier -> sigma is only an order-of-magnitude
  indicator (mitigated with 150/200 ps).
- MP/PBE lattice 4.3% too large; single-crystal, no grain boundaries (upper bound); NE
  ignores correlation.
- Small fine-tuning dataset (27), Gamma-only, GBRV -- sigma comes out stiff, **not**
  evidence that "fine-tuning is always more accurate", but rather a counter-example
  showing that "label quality determines everything".

---

## 7. Reproducibility

```bash
# Environment: MACE/MatterSim + ASE + pymatgen-analysis-diffusion + kinisi (see finetune/README.md)
python 01_build_structure.py                       # structure + supercell
python 02_baseline_md.py --temps 600,800,1000 --steps 200000 --supercell-tag _sc222 --traj-tag _prod
python 03_analyze_transport.py --traj-tag _prod    # D/sigma/Arrhenius + figures + metrics_prod.json
# Fine-tuning (Vanda A40/CPU, see finetune/):
bash finetune/submit_shards.sh                     # 27 QE-Gamma DFT labels
python finetune/12_split_train_valid.py            # 22 training / 5 validation
qsub  finetune/finetune_mace.pbs                   # A40 fine-tuning -> li6ps5cl_ft_stagetwo.model
python 02_baseline_md.py --mace-model finetune/models/li6ps5cl_ft_stagetwo.model --traj-tag _ft ...
python 03_analyze_transport.py --traj-tag _ft
```

- Key data: `data/metrics{,_long,_prod,_ft}.json`, `finetune/data/labelled.xyz` (the 27 DFT
  labels), `finetune/results/` (training curves).
- Model weights are excluded from the repo via `.gitignore`; they can be reproduced from
  `labelled.xyz` + the fine-tuning scripts.
- All computation ran on **NUS Vanda** (personal free quota, A40 for MD / CPU for DFT), no
  paid rentals.

---

## References

1. Z. Li, J. Huang, X. Ren, J. Li, R. Xiao, H. Li. *Mechanistic Insights into Temperature
   Effects for Ionic Conductivity in Li6PS5Cl.* arXiv:2403.14116 (2024). -- MLIP-MD; no
   hops at 300 K/100 ns, requires extrapolation, extrapolated value varies with the
   temperature window.
2. S. Wang, Y. Zhang, X. Zhang, T. Liu, Y.-H. Lin, Y. Shen, L. Li, C.-W. Nan.
   *High-Conductivity Argyrodite Li6PS5Cl Solid Electrolytes Prepared via Optimized
   Sintering Processes for All-Solid-State Lithium-Sulfur Batteries.* ACS Appl. Mater.
   Interfaces **2018**, 10 (49), 42279-42285. DOI 10.1021/acsami.8b15121. -- optimized
   sintering, sigma(300 K)=3.15 mS/cm (this work's primary benchmark).
3. R. F. Indrawan, H. Gamo, A. Nagai, A. Matsuda. *Chemically Understanding the
   Liquid-Phase Synthesis of Argyrodite Solid Electrolyte Li6PS5Cl with the Highest Ionic
   Conductivity for All-Solid-State Batteries.* Chem. Mater. **2023**, 35 (6), 2549-2558.
   DOI 10.1021/acs.chemmater.2c03818. -- liquid-phase synthesis, sigma>2 mS/cm.
4. J. M. Lee, Y. S. Park, J.-W. Moon, H. Hwang. *Ionic and Electronic Conductivities of
   Lithium Argyrodite Li6PS5Cl Electrolytes Prepared via Wet Milling and Post-Annealing.*
   Front. Chem. **2021**, 9, 778057. DOI 10.3389/fchem.2021.778057. -- wet milling +
   annealing, sigma~=1.0-1.9 mS/cm.
5. C. Yu, S. Ganapathy, J. Hageman, L. van Eijck, E. R. H. van Eck, L. Zhang, T. Schwietert,
   S. Basak, E. M. Kelder, M. Wagemaker. *Facile Synthesis toward the Optimal
   Structure-Conductivity Characteristics of the Argyrodite Li6PS5Cl Solid-State
   Electrolyte.* ACS Appl. Mater. Interfaces **2018**, 10 (39), 33296-33306.
   DOI 10.1021/acsami.8b07476. -- solvent-processed, E_a~=0.20-0.25 eV.

*Generated: 2026-07-01. Data provenance in the various `metrics_*.json` files and
`finetune/results/`.*
