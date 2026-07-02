# Figures — Li₆PS₅Cl MLIP-MD ionic conductivity (Project 2)

Publication figure set with legends + statistics/provenance, in the project's logical order. The repo
tracks **PNG (600 dpi)**; the editable **SVG + vector PDF** submission bundle (`svg.fonttype='none'`,
`pdf.fonttype=42`; no TIFF, that's for photos) is regenerated on demand via
`finalize_figure(..., formats=("png","svg","pdf"))`. Figures are drawn at
Nature column widths (single ≈89 mm, double ≈183 mm), 5–7 pt sans text, 8 pt bold lowercase panel
letters, **full 4-sided box frame**, **direct in-colour curve labels (no legend boxes)**, kinisi
error bars, restrained CVD-safe palette. Every quantitative figure has a `source_data/<name>.csv`.

## The story (why each figure exists)
System → does the MD conserve? → do untuned universal potentials get σ right? (they over-predict) →
is that a sampling artefact? (longer MD converges it down) → production-quality σ(300 K) → can DFT
fine-tuning improve the potential? → what does fine-tuning do to σ? Figures are numbered 01→04 along
this arc; the four `_tag` transport variants (baseline / `_long` / `_prod` / `_ft`) are the successive
steps of that argument.

---

## Main figures

**Fig. 1 | The Li₆PS₅Cl argyrodite model system.** `figures/01_structures.*` — *[system]*
(**a**) Ewald-lowest S/Cl ordering (ground state); (**b**) next-ranked variant. 52-atom conventional
cell (MP mp-985592), free-anion S²⁻/Cl⁻ disorder resolved by Ewald enumeration; spheres coloured by
element (Li/P/S/Cl legend), dashed unit cell. *Stats:* deterministic structures (none).
*Source:* `source_data/01_structures.csv`.

**Fig. 2 | Production conductivity (416-atom, foundation MACE-MP-0).** `figures/03_arrhenius_prod.*`
+ `figures/03_sigma300_vs_expt_prod.*` — *[headline result]*
Arrhenius plot on both axes ((**a**) log₁₀σ, curved; (**b**) log₁₀ σT, linear) + the σ(300 K) bar vs
experiment. 2×2×2 supercell, 200 ps, three temperatures. **Result: σ(300 K)=5.57 mS cm⁻¹ (1.8×
experiment), Eₐ=0.265 eV, R²=0.999** — near-literature agreement from a pure, un-fine-tuned MLIP.
*Stats:* n=1 trajectory per T; centre = pymatgen DiffusionAnalyzer D → Nernst-Einstein σ; error bars =
kinisi bootstrap 1σ on D (GLS, MSD autocorrelation) propagated to σ; Arrhenius = OLS of ln(σT) vs 1/T
(3 T). *Source:* `source_data/03_arrhenius_prod.csv`, `03_sigma300_vs_expt_prod.csv`.

**Fig. 3 | DFT fine-tuning of MACE-MP-0.** `figures/04_finetune_convergence.*` +
`figures/03_arrhenius_ft.*` + `figures/03_sigma300_vs_expt_ft.*` — *[fine-tuning + its effect]*
(**04 a,b**) validation force / energy RMSE vs epoch against the foundation baseline (dashed);
~epoch-90 spike = SWA stage-two restart. (**03_ft**) the fine-tuned model's Arrhenius + σ(300 K).
**Result: validation force RMSE 234→62 meV Å⁻¹; the fine-tuned σ(300 K)=0.497 mS cm⁻¹ (0.16×) —
i.e. the potential stiffened (Eₐ 0.265→0.331 eV), so foundation over-predicts and fine-tune
under-predicts, bracketing experiment.** *ML stats:* 27 QE-Γ-DFT (GBRV-PBE) snapshots, 22 train / 5
validation (seed 0); metric = held-out validation RMSE; baseline = MACE-MP-0 small (pre-fine-tune);
no test set (small-data delta fine-tune). *Source:* `source_data/04_finetune_convergence.csv`,
`03_arrhenius_ft.csv`, `03_sigma300_vs_expt_ft.csv`.

## Supplementary figures

**Fig. S1 | MD conservation check.** `figures/02_md_stability_mace{,_prod,_ft}.*` — *[method validation]*
(**a**) instantaneous temperature, (**b**) potential energy per atom vs time; direct-labelled 600/800/
1000 K traces, dashed = target T. Confirms stable NVT thermostatting, no drift/blow-up. Variants:
baseline (52-atom, 50 ps), `_prod` (416-atom, 200 ps), `_ft` (416-atom, 200 ps, fine-tuned; energies
on the fine-tuned QE reference ≈ −269 eV/atom). *Stats:* n=1 Langevin-NVT trajectory per T (friction
0.01 fs⁻¹, 1 fs step, log every 50). *Source:* `source_data/02_md_stability_mace{,_prod,_ft}.csv`.

**Fig. S2 | Baseline transport & sampling convergence.** `figures/03_arrhenius{,_long}.*` +
`figures/03_sigma300_vs_expt{,_long}.*` — *[baseline → convergence]*
Untuned MACE + MatterSim at 50 ps (baseline) and 150 ps (`_long`). Both universal potentials
over-predict single-crystal σ at 50 ps; extending to 150 ps converges them downward — MACE
29.6→12.9 mS cm⁻¹ (9.4×→4.1×, R² 0.81→0.999), **MatterSim 16.0→2.06 (5.1×→0.65×, R² 0.85→0.983,
the closest-to-experiment run)** — showing the short-run over-prediction is largely unconverged-MSD
statistics, not physics. *Stats:* as Fig. 2. *Source:* `source_data/03_arrhenius{,_long}.csv`,
`03_sigma300_vs_expt{,_long}.csv`.
