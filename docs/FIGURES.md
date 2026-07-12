# Figures — Li₆PS₅Cl MLIP-MD ionic conductivity (Project 2)

Publication figure set with legends + statistics/provenance, in the project's logical order. The repo
tracks **PNG (600 dpi)**; the editable **SVG + vector PDF** submission bundle (`svg.fonttype='none'`,
`pdf.fonttype=42`; no TIFF, that's for photos) is regenerated on demand via
`finalize_figure(..., formats=("png","svg","pdf"))`. Figures are drawn at
Nature column widths (single ≈89 mm, double ≈183 mm), 5–7 pt sans text, 8 pt bold lowercase panel
letters, **full 4-sided box frame**, **direct in-colour curve labels (no legend boxes)**, kinisi
error bars, restrained CVD-safe palette. Figures are tiered into `figures/main/` (the five headline
figures below) and `figures/supplementary/`; every quantitative figure has a matching
`source_data/<tier>/<name>.csv`. Paths below are relative to this `docs/` folder (`../figures/…`).

## How the files are named (read this first)

File names carry **provenance, not figure order**: the `NN_` prefix is the pipeline script that
draws the file, and the suffix names the convergence tier / composition / lead. The figure numbers
(Fig. 1–5, S1–S5) live only in this document. Map:

Every transport figure is one `03_transport…` file: panel **a** = the Arrhenius fit, panel
**b** = the σ(300 K) bars (a single analysis run), and the merged supplementary figures stack
one such row per tag / composition / lead (panels a, b, c, … row-major, row label top-right).

| file (`figures/<tier>/…`) | figure | tier | produced by |
|---|---|---|---|
| `main/01_structures.png` | Fig. 1 | — | `01_build_structure.py` |
| `main/03_transport_prod.png` | Fig. 2 | prod | `03_analyze_transport.py --traj-tag _prod` |
| `main/04_finetune_convergence.png` | Fig. 3 | ft | `finetune/21_plot_finetune.py` |
| `main/03_transport_ft.png` | Fig. 3 | ft | `03_analyze_transport.py --traj-tag _ft` |
| `main/07_doping_trend.png` | Fig. 4 | dope | `07_doping_trend.py` |
| `main/06_compare_leads.png` | Fig. 5 | lead | `06_compare_leads.py` |
| `supplementary/02_md_stability_<mlip><tag>.png` (×4) | Fig. S1 | all | `02_baseline_md.py` |
| `supplementary/03_transport_convergence.png` | Fig. S2 | baseline, long | `03_analyze_transport.py --merge-tags ",_long" --merge-name convergence` |
| `supplementary/03_transport_dope.png` | Fig. S3 | dope | `03_analyze_transport.py --merge-tags "_dope_cl100,…,_dope_cl175" --merge-name dope --merge-layout overlay` |
| `supplementary/03_transport_leads.png` | Fig. S4 | lead | `03_analyze_transport.py --merge-tags "_lead_<key>,…" --merge-name leads` |
| `supplementary/08_pipeline_overview.{png,svg}` | Fig. S5 | — | `08_pipeline_overview.py` |

**Convergence tiers** (the `--traj-tag` suffix; one mental model for every transport figure):

| tag | cell × duration | role |
|---|---|---|
| *(none)* | 52-atom × 50 ps | thin baseline — MSD **not** converged, wide bands |
| `_long` | 52-atom × 150 ps | convergence check |
| `_prod` | **416-atom × 200 ps** | **production headline** (σ = 5.55 mS/cm) |
| `_ft` | 416-atom × 200 ps | fine-tuned MACE, same protocol as `_prod` |
| `_dope_cl<NNN>` | 392–416-atom × 150 ps | screening-grade, Cl-excess series |
| `_lead_<key>` | 96–200-atom × 150 ps | screening-grade, W11 funnel leads |

## The story (why each figure exists)
System → does the MD conserve? → do untuned universal potentials get σ right? (they over-predict) →
is that a sampling artefact? (longer MD converges it down) → production-quality σ(300 K) → can DFT
fine-tuning improve the potential? → what does fine-tuning do to σ? → do the upstream funnel leads
survive the same protocol? (the W11 rank reversal, Fig. 5). The `_tag` transport variants
(baseline / `_long` / `_prod` / `_ft` / `_dope` / `_lead`) are the successive steps of that argument.

---

## Main figures

**Fig. 1 | The Li₆PS₅Cl argyrodite model system.** `../figures/main/01_structures.*` — *[system]*
(**a**) Ewald-lowest S/Cl ordering (ground state); (**b**) next-ranked variant. 52-atom conventional
cell (MP mp-985592), free-anion S²⁻/Cl⁻ disorder resolved by Ewald enumeration; spheres coloured by
element (Li/P/S/Cl legend), dashed unit cell. *Stats:* deterministic structures (none).
*Source:* `../source_data/main/01_structures.csv`.

**Fig. 2 | Production conductivity (416-atom, foundation MACE-MP-0 vs MatterSim).**
`../figures/main/03_transport_prod.*` — *[headline result]*
(**a**) Arrhenius fit; (**b**) σ(300 K) bars vs the two experimental references.
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
`../source_data/main/03_transport_prod.csv`.

**Fig. 3 | DFT fine-tuning of MACE-MP-0.** `../figures/main/04_finetune_convergence.*` +
`../figures/main/03_transport_ft.*` — *[fine-tuning + its effect]*
(**04 a,b**) validation force / energy RMSE vs epoch against the foundation baseline (dashed);
~epoch-90 spike = SWA stage-two restart. (**03_transport_ft a,b**) the fine-tuned model's
Arrhenius fit + σ(300 K) bars.
**Result: validation force RMSE 234→62 meV Å⁻¹; the fine-tuned σ(300 K)=0.45 [0.31, 0.66] mS cm⁻¹
(0.14×) — i.e. the potential stiffened (Eₐ 0.265→0.335 eV), so foundation over-predicts and fine-tune
under-predicts, bracketing experiment (and robustly so: the production [4.2, 7.3] and fine-tuned
[0.31, 0.66] bands both exclude experiment, from opposite sides).** *ML stats:* 27 QE-Γ-DFT (GBRV-PBE)
snapshots, 22 train / 5
validation (seed 0); metric = held-out validation RMSE; baseline = MACE-MP-0 small (pre-fine-tune);
no test set (small-data delta fine-tune). *Source:* `../source_data/main/04_finetune_convergence.csv`,
`03_transport_ft.csv`.

**Fig. 4 | Cl-excess doping trend.** `../figures/main/07_doping_trend.*` — *[composition-property trend]*
(**a**) σ(300 K) vs Cl content in Li₆₋ₓPS₅₋ₓCl₁₊ₓ (x=0/0.25/0.5/0.75), log y — ~29× monotonic
increase. (**b**) Eₐ vs Cl content, linear y — monotonic decrease (0.256→0.148 eV). Same MACE-MP-0
(small) / NVT Langevin / 600–800–1000 K / 150 ps protocol as the W11 funnel leads (Fig. S2-tier, not
the 416-atom/200 ps production tier of Fig. 2) — the Cl=1.0 anchor here (7.31 mS cm⁻¹) is therefore
not the production Li₆PS₅Cl baseline (5.55 mS cm⁻¹); same order of magnitude, different convergence
tier, not a discrepancy. *Stats:* as Fig. 2 (kinisi bootstrap error bars, weighted Arrhenius fit, 3 T per
composition). *Source:* `../source_data/main/07_doping_trend.csv`.

**Fig. 5 | W11 funnel handoff — four upstream leads under the same MD protocol.**
`../figures/main/06_compare_leads.*` — *[funnel validation]*
(**a**) Arrhenius fits of the four leads handed down by the upstream funnel (Project 1 screen:
Li₂₀Si₃P₃S₂₃Cl, Li₈TiS₆; Project 3 generation: Li₃PS₄ gen016, LiPS₃ gen021), with the Li₆PS₅Cl
production baseline dashed; (**b**) σ(300 K) bars annotated with each lead's upstream prior rank.
**Result: rank reversal — LiPS₃, ranked *last* (4th) by the coarse prior, certifies as the
2nd-strongest conductor (10 [5.4, 20] mS cm⁻¹, Eₐ = 0.243 eV), while 2nd-ranked Li₈TiS₆ is falsified
as near-insulating (~9×10⁻⁴ mS cm⁻¹). Two of four leads survive, one from each upstream source —
the funnel screens by the actual chemistry, not by which upstream method was "right".**
Screening-grade certification (150 ps tier, `--no-expt`: no experimental σ exists for these leads);
the near-insulating leads carry wide bands but sit orders of magnitude below the survivors with no
overlap, so the verdicts are unambiguous. *Stats:* as Fig. 2 (kinisi bootstrap, weighted Arrhenius).
*Source:* `../source_data/main/06_compare_leads.csv`.

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

**Fig. S2 | Baseline transport & sampling convergence.**
`../figures/supplementary/03_transport_convergence.*` — *[baseline → convergence]*
One row per sampling tier — (**a,b**) 52-atom / 50 ps, (**c,d**) 52-atom / 150 ps; left =
Arrhenius, right = σ(300 K) bars.
Untuned MACE + MatterSim at 50 ps (baseline) and 150 ps (`_long`). Both universal potentials
over-predict single-crystal σ at 50 ps; extending to 150 ps converges them downward — MACE
22.7→12.5 mS cm⁻¹ (7.2×→4.0×, R² 0.79→0.999), **MatterSim 14.1→1.79 (4.5×→0.57×, R² 0.83→0.982,
the closest single run to experiment)** — showing the short-run over-prediction is largely unconverged-MSD
statistics, not physics. *Stats:* as Fig. 2. *Source:*
`../source_data/supplementary/03_transport_convergence.csv`.

**Fig. S3 | Per-composition transport, Cl-excess series.**
`../figures/supplementary/03_transport_dope.*` — *[Fig. 4 variants]*
(**a**) The four single-composition Arrhenius fits (log₁₀ σT vs 1000/T) underlying Fig. 4's
trend, overlaid in one panel — one series colour per Cl content, legend carries each fit's Eₐ
(0.26 → 0.15 eV), so the flattening slope with Cl excess reads directly; (**b**) the four
σ(300 K) bars in one shared log axis — the ~29× monotonic rise (7.3 → 213 mS cm⁻¹).
(`--no-expt`: no per-composition experimental reference exists for the non-anchor
compositions; colours match Fig. 4's composition ramp.) *Stats:* as Fig. 2. *Source:*
`../source_data/supplementary/03_transport_dope.csv`.

**Fig. S4 | Per-lead transport, W11 funnel leads.**
`../figures/supplementary/03_transport_leads.*` — *[Fig. 5 variants]*
One row per lead, in upstream prior order — (**a,b**) Li₂₀Si₃P₃S₂₃Cl, (**c,d**) Li₈TiS₆,
(**e,f**) Li₃PS₄ gen016, (**g,h**) LiPS₃ gen021; left = Arrhenius fit, right = σ(300 K) bar
(`--no-expt`). Weighted-fit values: Li₂₀Si₃P₃S₂₃Cl **29 [18, 46]** mS cm⁻¹ (Eₐ 0.199 eV, R² 0.994);
LiPS₃ **10 [5.4, 20]** (0.243, 0.996); Li₃PS₄ gen016 5.5×10⁻³ [2×10⁻³, 1.5×10⁻²] (0.545, R² 0.87);
Li₈TiS₆ 9.0×10⁻⁴ [3.3×10⁻⁴, 2.5×10⁻³] (0.602, R² 0.77). The two near-insulators fit poorly (few
uncorrelated hops even at 600 K → wide bands, low R²) but sit orders of magnitude below the
survivors, so the verdicts are unambiguous. *Stats:* as Fig. 2. *Source:*
`../source_data/supplementary/03_transport_leads.csv`.

**Fig. S5 | Portfolio pipeline overview.**
`../figures/supplementary/08_pipeline_overview.{png,svg}` — *[context]*
Schematic of the three-project funnel: Project 1 (CatBoost screen + adversarial audit, 328 → 184)
and Project 3 (MatterGen + self-consistent MLIP hull, 64 → 43 S.U.N.) each hand leads to this
repo's same-protocol MLIP-MD validation, with the future electrochemical closure (EIS) dashed.
The rank-reversal callouts quote Fig. 5. All box numbers are hardcoded literals with source-file
comments (`08_pipeline_overview.py`) so the figure renders standalone; it is embedded as the
banner of all three portfolio repo READMEs. *Stats:* schematic (none). *Source:*
`../source_data/supplementary/08_pipeline_overview.csv`.
