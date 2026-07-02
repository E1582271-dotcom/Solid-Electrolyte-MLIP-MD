# Figure legends & statistics (Project 2 — Li₆PS₅Cl MLIP-MD)

Submission-grade figure captions + statistics/provenance, per the Nature figure QA contract.
Every figure ships as **PNG (600 dpi preview) + SVG + PDF (editable vector, `pdf.fonttype=42` /
`svg.fonttype='none'`)** at Nature column widths (single ≈89 mm, double ≈183 mm), 5–7 pt sans text,
8 pt bold lowercase panel letters, top/right spines off, frameless/direct legends. Line-art graphs use
vector PDF (Nature's preferred format for graphs); no TIFF (reserved for photographic panels — none here).
Each quantitative figure has a `source_data/<name>.csv`.

---

**Fig. 1 | Ordered S/Cl approximants of the Li₆PS₅Cl argyrodite cell.** `figures/01_structures.*`
(**a**) Ewald-lowest ordering (ground state); (**b**) next-ranked variant. 52-atom conventional cell
from Materials Project mp-985592 with the S²⁻/Cl⁻ free-anion disorder resolved by Ewald-energy
enumeration. Spheres coloured by element (Li/P/S/Cl legend, ASE jmol colours); dashed box = unit cell.
*Statistics:* none (deterministic ordered structures). *Source data:* `source_data/01_structures.csv`
(per config: element counts, Ewald energy in eV).

**Fig. 2 | NVT-MD conservation check.** `figures/02_md_stability_mace{,_prod,_ft}.*`
(**a**) instantaneous temperature vs time; (**b**) potential energy per atom vs time, for target
T = 600/800/1000 K (dashed = target T). Confirms stable thermostatting with no energy drift/blow-up.
Variants: baseline (52-atom, 50 ps, MACE-MP-0), `_prod` (416-atom 2×2×2, 200 ps), `_ft` (416-atom,
200 ps, fine-tuned model; energies on the fine-tuned QE reference scale ≈ −269 eV/atom).
*Statistics:* n = 1 Langevin-NVT trajectory per temperature (friction 0.01 fs⁻¹, 1 fs step, log every
50 steps); traces are instantaneous per-frame values, not averages. *Source data:*
`source_data/02_md_stability_mace{,_prod,_ft}.csv` (per T: mean_T, std_T, mean_E/atom, energy drift in
meV atom⁻¹ ps⁻¹).

**Fig. 3 | Li⁺ conductivity: Arrhenius analysis.** `figures/03_arrhenius{,_long,_prod,_ft}.*`
(**a**) log₁₀ σ vs 1000/T (curved: σ = σT/T); (**b**) log₁₀(σT) vs 1000/T (the linear Arrhenius form
actually fitted). Filled circles = MD, line = Arrhenius fit, star = 300 K extrapolation, ✕ = experiment;
dashed line / dotted vertical = experiment (3.15 mS cm⁻¹) and 300 K. Corner box gives Eₐ, σ₃₀₀, R².
Variants: baseline (MACE + MatterSim, 52-atom 50 ps), `_long` (both, 150 ps), `_prod` (MACE, 416-atom
200 ps), `_ft` (fine-tuned MACE, 416-atom 200 ps).
*Statistics:* n = 1 trajectory per (potential, temperature). Centre = Li⁺ diffusivity D from the
pymatgen `DiffusionAnalyzer` (MSD backbone) → Nernst-Einstein σ. **Error bars = kinisi bootstrap 1σ
uncertainty on D (generalised-least-squares fit accounting for MSD autocorrelation), propagated to σ on
the log axis.** Arrhenius fit = ordinary least squares of ln(σT) vs 1/T over the 3 temperatures; R²
reported. σ₃₀₀ is the 300 K extrapolation of that fit. *Source data:*
`source_data/03_arrhenius{tag}.csv` (mlip, T, σ, kinisi σ std, fit Eₐ/σ₃₀₀/R²).

**Fig. 3 (bar) | Extrapolated σ(300 K) vs experiment.** `figures/03_sigma300_vs_expt{,_long,_prod,_ft}.*`
Bars: simulated σ(300 K) (Arrhenius extrapolation) vs experimental Li₆PS₅Cl (sintered 3.15, mechanochemical
1.33 mS cm⁻¹). Log y-axis. *Statistics:* bars are point estimates (the Fig. 3 Arrhenius extrapolation);
per-temperature statistical uncertainty is shown in Fig. 3. *Source data:*
`source_data/03_sigma300_vs_expt{tag}.csv`.

**Fig. 4 | MACE-MP-0 → Li₆PS₅Cl fine-tuning convergence.** `figures/04_finetune_convergence.*`
(**a**) validation force RMSE vs epoch; (**b**) validation energy RMSE vs epoch. Dashed line = foundation
(pre-fine-tune) baseline; the ~epoch-90 spike is the SWA stage-two restart. *ML statistics:* 27 QE-DFT
(Γ-only, GBRV-PBE) labelled snapshots split 22 train / 5 validation (fixed seed 0); metric = RMSE on the
held-out validation set; baseline = the MACE-MP-0 small foundation model before fine-tuning
(234 meV Å⁻¹ / 118 meV atom⁻¹); final = 65 meV Å⁻¹ / 4 meV atom⁻¹ (SWA stage-two model, reported
62.4 meV Å⁻¹). No separate test set (small-data delta fine-tune). *Source data:*
`source_data/04_finetune_convergence.csv` (epoch, validation force/energy RMSE).
