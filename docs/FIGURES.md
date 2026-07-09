# Figures — Li₆PS₅Cl MLIP-MD ionic conductivity (Project 2)

Publication figure set with legends + statistics/provenance, in the project's logical order. The repo
tracks **PNG (600 dpi)**; the editable **SVG + vector PDF** submission bundle (`svg.fonttype='none'`,
`pdf.fonttype=42`; no TIFF, that's for photos) is regenerated on demand via
`finalize_figure(..., formats=("png","svg","pdf"))`. Figures are drawn at
Nature column widths (single ≈89 mm, double ≈183 mm), 5–7 pt sans text, 8 pt bold lowercase panel
letters, **full 4-sided box frame**, **direct in-colour curve labels (no legend boxes)**, kinisi
error bars, restrained CVD-safe palette. Figures are tiered into `figures/main/` (the four headline
figures below) and `figures/supplementary/`; every quantitative figure has a matching
`source_data/<tier>/<name>.csv`. Paths below are relative to this `docs/` folder (`../figures/…`).

## The story (why each figure exists)
System → does the MD conserve? → do untuned universal potentials get σ right? (they over-predict) →
is that a sampling artefact? (longer MD converges it down) → production-quality σ(300 K) → can DFT
fine-tuning improve the potential? → what does fine-tuning do to σ? Figures are numbered 01→04 along
this arc; the four `_tag` transport variants (baseline / `_long` / `_prod` / `_ft`) are the successive
steps of that argument.

---

## Main figures

**Fig. 1 | The Li₆PS₅Cl argyrodite model system.** `../figures/main/01_structures.*` — *[system]*
(**a**) Ewald-lowest S/Cl ordering (ground state); (**b**) next-ranked variant. 52-atom conventional
cell (MP mp-985592), free-anion S²⁻/Cl⁻ disorder resolved by Ewald enumeration; spheres coloured by
element (Li/P/S/Cl legend), dashed unit cell. *Stats:* deterministic structures (none).
*Source:* `../source_data/main/01_structures.csv`.

**Fig. 2 | Production conductivity (416-atom, foundation MACE-MP-0 vs MatterSim).**
`../figures/main/03_arrhenius_prod.*` + `../figures/main/03_sigma300_vs_expt_prod.*` — *[headline result]*
Arrhenius plot in the linear form (log₁₀ σT vs 1000/T — points=MD, line=fit, ★=300 K extrapolation,
✕=experiment) + the σ(300 K) bars vs experiment, one pair of curves/bars per potential. 2×2×2
supercell, 200 ps, three temperatures, both potentials. **Result: MACE σ(300 K)=5.55 [4.2, 7.3] mS cm⁻¹
(1.8× experiment), Eₐ=0.265 eV, R²=0.999 — near-literature agreement from a pure, un-fine-tuned MLIP.
MatterSim σ(300 K)=0.35 [0.25, 0.49] mS cm⁻¹ (0.11×), Eₐ=0.346 eV, R²=0.991 — under-predicts, and more
severely than its own 52-atom/150 ps run (0.57×, the closest single run to experiment in Fig. S2). The two
potentials sit on opposite sides of experiment at both convergence tiers, but their bias moves in
opposite directions with better sampling: MACE's over-prediction shrinks toward experiment
(7.2×/4.0× → 1.8×, Fig. S2 → Fig. 2), while MatterSim's under-prediction grows worse (0.57× →
0.11×) — more atoms/longer time converges MACE toward the truth but pulls MatterSim away from it.**
*Stats:* n=1 trajectory per T per potential; centre = pymatgen DiffusionAnalyzer D → Nernst-Einstein
σ; error bars = kinisi bootstrap 1σ on D (GLS, MSD autocorrelation) propagated to σ; Arrhenius =
**weighted** least squares of ln(σT) vs 1/T (each T weighted by its kinisi σ uncertainty, Mo-group `aimd`
standard), σ(300 K) reported with the propagated [min, max] interval. *Source:*
`../source_data/main/03_arrhenius_prod.csv`, `03_sigma300_vs_expt_prod.csv`.

**Fig. 3 | DFT fine-tuning of MACE-MP-0.** `../figures/main/04_finetune_convergence.*` +
`../figures/main/03_arrhenius_ft.*` + `../figures/main/03_sigma300_vs_expt_ft.*` — *[fine-tuning + its effect]*
(**04 a,b**) validation force / energy RMSE vs epoch against the foundation baseline (dashed);
~epoch-90 spike = SWA stage-two restart. (**03_ft**) the fine-tuned model's Arrhenius + σ(300 K).
**Result: validation force RMSE 234→62 meV Å⁻¹; the fine-tuned σ(300 K)=0.45 [0.31, 0.66] mS cm⁻¹
(0.14×) — i.e. the potential stiffened (Eₐ 0.265→0.335 eV), so foundation over-predicts and fine-tune
under-predicts, bracketing experiment (and robustly so: the production [4.2, 7.3] and fine-tuned
[0.31, 0.66] bands both exclude experiment, from opposite sides).** *ML stats:* 27 QE-Γ-DFT (GBRV-PBE)
snapshots, 22 train / 5
validation (seed 0); metric = held-out validation RMSE; baseline = MACE-MP-0 small (pre-fine-tune);
no test set (small-data delta fine-tune). *Source:* `../source_data/main/04_finetune_convergence.csv`,
`03_arrhenius_ft.csv`, `03_sigma300_vs_expt_ft.csv`.

**Fig. 4 | Cl-excess doping trend.** `../figures/main/07_doping_trend.*` — *[composition-property trend]*
(**a**) σ(300 K) vs Cl content in Li₆₋ₓPS₅₋ₓCl₁₊ₓ (x=0/0.25/0.5/0.75), log y — ~29× monotonic
increase. (**b**) Eₐ vs Cl content, linear y — monotonic decrease (0.256→0.148 eV). Same MACE-MP-0
(small) / NVT Langevin / 600–800–1000 K / 150 ps protocol as the W11 funnel leads (Fig. S2-tier, not
the 416-atom/200 ps production tier of Fig. 2) — the Cl=1.0 anchor here (7.31 mS cm⁻¹) is therefore
not the production Li₆PS₅Cl baseline (5.55 mS cm⁻¹); same order of magnitude, different convergence
tier, not a discrepancy. *Stats:* as Fig. 2 (kinisi bootstrap error bars, weighted Arrhenius fit, 3 T per
composition). *Source:* `../source_data/supplementary/07_doping_trend.csv`.

## Supplementary figures

**Fig. S1 | MD conservation check.**
`../figures/supplementary/02_md_stability_mace{,_prod,_ft}.*` + `../figures/supplementary/02_md_stability_mattersim_prod.*` —
*[method validation]*
(**a**) instantaneous temperature, (**b**) potential energy per atom vs time; direct-labelled 600/800/
1000 K traces, dashed = target T. Confirms stable NVT thermostatting, no drift/blow-up, for **both**
production-tier potentials (MACE and MatterSim, 416-atom/200 ps) as well as the 52-atom baseline and
fine-tuned variants. Variants: baseline (52-atom, 50 ps, MACE), `_prod` (416-atom, 200 ps, MACE and
MatterSim), `_ft` (416-atom, 200 ps, fine-tuned MACE; energies on the fine-tuned QE reference ≈ −269
eV/atom). *Stats:* n=1 Langevin-NVT trajectory per T per potential (friction 0.01 fs⁻¹, 1 fs step, log
every 50). *Source:* `../source_data/supplementary/02_md_stability_mace{,_prod,_ft}.csv`,
`../source_data/supplementary/02_md_stability_mattersim_prod.csv`.

**Fig. S2 | Baseline transport & sampling convergence.** `../figures/supplementary/03_arrhenius{,_long}.*` +
`../figures/supplementary/03_sigma300_vs_expt{,_long}.*` — *[baseline → convergence]*
Untuned MACE + MatterSim at 50 ps (baseline) and 150 ps (`_long`). Both universal potentials
over-predict single-crystal σ at 50 ps; extending to 150 ps converges them downward — MACE
22.7→12.5 mS cm⁻¹ (7.2×→4.0×, R² 0.79→0.999), **MatterSim 14.1→1.79 (4.5×→0.57×, R² 0.83→0.982,
the closest single run to experiment)** — showing the short-run over-prediction is largely unconverged-MSD
statistics, not physics. *Stats:* as Fig. 2. *Source:* `../source_data/supplementary/03_arrhenius{,_long}.csv`,
`03_sigma300_vs_expt{,_long}.csv`.

**Fig. S3 | Per-composition Arrhenius, Cl-excess series.**
`../figures/supplementary/03_arrhenius_dope_cl{100,125,150,175}.*` + `../figures/supplementary/03_sigma300_dope_cl{100,125,150,175}.*`
— *[Fig. 4 variants]* The four single-composition Arrhenius fits (log₁₀ σT vs 1000/T) and σ(300 K)
bars underlying Fig. 4's trend, one per Cl content (`--no-expt`: no per-composition experimental
reference exists for the non-anchor compositions). *Stats:* as Fig. 2. *Source:*
`../source_data/supplementary/03_arrhenius_dope_cl{100,125,150,175}.csv`, `03_sigma300_dope_cl{100,125,150,175}.csv`.
