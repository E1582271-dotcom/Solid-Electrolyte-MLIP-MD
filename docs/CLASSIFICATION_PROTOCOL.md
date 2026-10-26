# Definitive classification protocol for the 62 census compositions

Status: **pre-registered on 2026-10-24, before any new MD is run and before the
formal re-classification is computed.** Rules, thresholds and the compute list
below are fixed by this document. Any later change must be recorded as a dated
amendment at the end of this file, with its reason, and must not be made after
seeing the outcome it would affect.

## Why

The seed-0 census (62 compositions × 600/800/1000 K × 200 ps, MACE-MP-0 small)
labelled 56 compositions "not reliable" because any one temperature failing a
quality gate removed the whole composition. That label mixed three different
situations: Li that does not diffuse (an informative negative), frameworks that
melt at 800/1000 K (a protocol-induced false negative), and 600 K runs too short
to resolve diffusion. This protocol replaces the composition-level veto with
per-temperature eligibility and assigns every composition exactly one terminal
class. Target: **0 compositions without a definitive class.**

## Definitions

- **Eligible temperature**: a (composition, T, seed) run whose Li diffusion is
  resolved and whose framework is intact, using the existing census criteria
  unchanged:
  - diffusion resolved: kinisi D lower bound > 0, Li MSD growth over the fit
    region ≥ 1 Å², log–log MSD slope within 0.5–1.5;
  - framework intact: non-Li MSD at 100 ps lag ≤ 1 Å² and fraction of non-Li
    atoms displaced > 2 Å ≤ 0.10, evaluated at a fixed 200 ps time so runs of
    different length are comparable;
  - mean temperature within 5 % of the target.
  Melted or unresolved temperatures are excluded from fits; they no longer veto
  the composition.
- **σ300**: Nernst–Einstein conductivity extrapolated to 300 K from an Arrhenius
  fit of ln D against 1/T over eligible temperatures, using the composition's
  n_Li and cell volume. All values are conditional on the un-fine-tuned
  MACE-MP-0 small potential and the MP/PBE volume.
- **Fast threshold**: σ300 ≥ **0.1 mS/cm**.
- **D upper bound** for a non-resolved run: D ≤ MSD_Li(100 ps)/(6 × 100 ps).
  This is conservative because the MSD includes the vibrational plateau. The
  kinisi upper credible bound is **not** used as a bound (it can fall below the
  direct-MSD estimate for non-diffusive runs).
- **Prefactor rule**: assume D₀ ≥ **1×10⁻⁴ cm²/s**, i.e. a²ν/6 with a ≈ 2.5 Å and
  ν ≈ 10¹² s⁻¹. A D upper bound at temperature T then implies
  Eₐ ≥ k_B·T·ln(D₀,min / D_bound), which bounds σ300 from above. The fitted D₀ of
  every composition with ≥ 2 eligible temperatures is reported to show the
  assumption is conservative. Sensitivity: every SLOW_BOUNDED verdict is also
  evaluated with D₀,min = 1×10⁻⁵ cm²/s and the result is reported.

## Terminal classes (exactly one per composition)

| Class | Rule |
|---|---|
| FAST_VALIDATED | Replicate rule passes (≥ 3 usable seeds at each fitted temperature, between-seed D max/min ≤ 2, per-seed and pooled Arrhenius R² ≥ 0.95; the documented Li4Re6S11 seeds-0/1 exception stands) **and** σ300 ≥ 0.1 mS/cm |
| SLOW_MEASURED | Arrhenius fit over ≥ 3 eligible temperatures (single- or multi-seed) with σ300 < 0.1 mS/cm |
| SLOW_ESTIMATED | Exactly 2 eligible temperatures with two-point σ300 < 0.1 mS/cm. If a lower temperature was tested and not resolved, the true Eₐ is higher and the estimate is an upper value |
| SLOW_BOUNDED | Fewer than 2 eligible temperatures and the prefactor-rule σ300 upper bound < 0.1 mS/cm |
| THERMALLY_UNSTABLE | Framework not intact at any tested temperature down to 400 K |

A composition that cannot be placed by existing data is sent to the compute list.
Terminal rule for compositions entering replicate validation: if the replicate
rule fails, add seeds up to 5 and extend runs up to 800 ps where needed; then fit
all seeds pooled. Pooled σ300 95 % lower bound ≥ 0.1 → FAST_VALIDATED; 95 % upper
bound < 0.1 → SLOW_MEASURED; otherwise classify by the pooled median and flag
`near_threshold`. A composition still unresolved at its lowest temperature after
800 ps is classified with the bound rules above.

## Compute list (NSCC ASPIRE 2A, A100-40GB; same potential, NVT, MP/PBE volume)

Batch 1, seed 0:
- Li20Si3P3S23Cl, Li5P(S2Cl)2, Li10Zn7(PS4)8, Li3NbS4, Li5BiS4 (framework melts
  at 800 K but Li is fast at 600 K): add 450 and 525 K, 400 ps production, to
  give three eligible temperatures with the existing 600 K run.
- Li48P16S61 (framework not intact at 600 K): 400, 450 and 500 K, 400 ps.
- Li9Nd2(PS4)5, Li6AsS5I (two-point σ300 ≥ 0.1 mS/cm from 600/800 K): add 700 K,
  200 ps.

Batch 2: every composition with σ300 ≥ 0.1 mS/cm after batch 1 receives seeds 1
and 2 at its three eligible temperatures, with the same run length as its seed-0
runs at those temperatures.

Raw trajectories stay on NSCC scratch; analysis runs on the cluster with the same
code; only results and remote trajectory sha256 values are downloaded.

## Precedents

- Excluding compositions whose framework melts during MD, and lowering the
  temperature window for low-melting chemistries: S. Wang et al., "Design
  principles for sodium superionic conductors", *Nat. Commun.* (2023),
  doi:10.1038/s41467-023-43436-3.
- Treating no diffusion at 1000 K as a non-conductor, and using longer runs where
  100–200 ps cannot resolve diffusion: L. Kahle, A. Marcolongo, N. Marzari,
  *Energy Environ. Sci.* 13, 928 (2020); Thakur, Ercole & Marzari, *Energy
  Environ. Sci.* (2026).
- Systematic potential-energy-surface softening of universal MLIPs (MACE-MP-0
  among them), which inflates diffusion and lowers simulated melting points:
  B. Deng et al., arXiv:2405.07105.

## Amendments

### A1 — 2026-10-24, before any new MD result exists

Applying the rules above to existing data (stage S2) changed the projected
compute list but not the rules:

1. **Li2SiSnS4 added to batch 1** (450 and 525 K, 400 ps). Only 600 K is
   eligible (800 K framework MSD 1.053 Å² > 1; 1000 K melted), and the MSD-bound
   σ300 upper bound is 0.27 mS/cm ≥ 0.1, so existing data cannot place it. The
   kinisi point D (1.3e-9 cm²/s) is not used as a bound, as stated above.
2. **Terminal rule applied to the two leads that failed replicate validation**:
   Li5SbS3I2 and Li3BiS3 receive seeds 3 and 4 at 600, 800 and 1000 K (200 ps at
   800/1000 K; 800 ps at 600 K, where earlier seeds were unresolved or
   window-sensitive), then are classified by the pooled rule. Pre-compute
   values: Li5SbS3I2 pooled σ300 0.0027 mS/cm (95 % CI 0.0003–0.021);
   Li3BiS3 pooled σ300 0.070 mS/cm (95 % CI 0.021–0.237, straddles 0.1).
3. **Implementation clarifications** (conservative readings, fixed now):
   (a) the prefactor rule is applied to every framework-intact temperature,
   always with the MSD(100 ps)/(6·100 ps) bound; (b) a slope bound from one
   eligible temperature plus a lower-temperature bound uses the smaller of the
   kinisi lower D and the direct-MSD D at the eligible temperature;
   (c) multi-seed fits pool all eligible runs (OLS of ln D on 1/T) and report the
   95 % interval of σ300.
4. **D₀ report**: fitted D₀ over the 15 compositions with ≥ 2 eligible
   temperatures has minimum 3.5×10⁻⁵ cm²/s (Li6AsS5I, two-point) and median
   5.7×10⁻³ cm²/s. No SLOW_BOUNDED verdict changes when D₀,min is lowered to
   1×10⁻⁵ cm²/s.

### A2 — 2026-10-24, after batch 1, before batch 2 is run

Batch 1 (29 runs, all raw-validated) was classified with the unchanged rules.
Li5SbS3I2 (5 seeds, σ300 0.0032 mS/cm) and Li3BiS3 (5 seeds, 600 K at 800 ps,
pooled σ300 0.090 mS/cm, flagged `near_threshold`) reached the terminal rule and
are final. Batch 2 applies the existing rules to the remaining compositions:

1. **Replicates (seeds 1 and 2) at the three eligible temperatures, with the
   seed-0 run lengths**: Li20Si3P3S23Cl, Li5P(S2Cl)2, Li10Zn7(PS4)8 and Li3NbS4
   (450 and 525 K at 400 ps; 600 K at 200 ps), and Li9Nd2(PS4)5 (600, 700 and
   800 K at 200 ps). All have seed-0 σ300 ≥ 0.1 mS/cm.
2. **Third eligible temperature plus replicates** for compositions with σ300 ≥
   0.1 mS/cm from only two eligible temperatures: Li48P16S61 adds 425 K (between
   its two framework-intact temperatures, 400 and 450 K; 500 K melted), then
   seeds 1 and 2 at 400, 425 and 450 K, all 400 ps. Li6AsS5I's 700 K seed-0 run
   was not resolved at 200 ps, so per the terminal rule its 700 K runs use 800 ps:
   seed 0 at 700 K (800 ps), then seeds 1 and 2 at 600 K (200 ps), 700 K (800 ps)
   and 800 K (200 ps).
3. **Sensitivity-triggered run**: Li2SiSnS4 is SLOW_BOUNDED at D₀,min = 1×10⁻⁴
   (σ300 ≤ 0.097 mS/cm) but not at 1×10⁻⁵, and its only eligible point (600 K,
   200 ps) has kinisi and direct-MSD D differing by ~300×. Seed 0 at 600 K is
   re-run at 800 ps; the longer run supersedes the 200 ps run, so the
   D₀-independent slope bound against its 525 K bound can decide it.
4. If any batch-2 composition fails the replicate rule, the terminal rule applies
   (seeds up to 5, runs up to 800 ps, then pooled classification).

### A3 — 2026-10-25, after batch 2, before batch 3 is run

Batch 2 (45 runs, all raw-validated) was classified with the A2 code. Two problems
were found **after seeing the batch-2 results**, so both changes below are post hoc.
Each is applied to all 62 compositions, and the affected verdicts are reported with and
without it. Earlier campaign manifests record the protocol sha256 of their own time.

1. **Implementation correction: the replicate rule comes before the slow class.** The
   classifier returned SLOW_MEASURED for any fit over ≥ 3 eligible temperatures with
   σ300 < 0.1 mS/cm before it evaluated the replicate rule. For compositions with
   replicate seeds this contradicts the terminal rule and A2.4, and it is not how A1.2
   treated Li5SbS3I2 and Li3BiS3 (sent to the terminal rule although their pooled σ300
   was below 0.1). The code now evaluates the replicate rule first whenever the eligible
   runs come from more than one seed. Pass: FAST_VALIDATED or SLOW_MEASURED by σ300.
   Fail: the terminal rule. Single-seed fits are unchanged. Effect: Li10Zn7(PS4)8 moves
   from SLOW_MEASURED to the terminal rule. Li5SbS3I2 and Li3BiS3 are at the terminal
   limit and keep their classes. On the pre-compute data the corrected code reproduces
   the A1 compute list (the pre-registered eight, Li2SiSnS4, Li5SbS3I2 and Li3BiS3).
2. **Failed kinisi fits.** Li10Zn7(PS4)8 450 K seed 1 meets the census eligibility
   criteria, but its kinisi D (5.1×10⁻¹⁰ cm²/s, covariance condition at the 10⁸ cap) is
   4,200× below the direct-MSD D of the same trajectory (2.1×10⁻⁶) and below the kinisi
   D of its first and second halves (2.5 and 2.9×10⁻⁶). This one run moves the pooled
   σ300 from 5.3 to 0.0012 mS/cm. Rule: an otherwise eligible run whose kinisi D differs
   from its direct-MSD D by more than 3× (the existing audit threshold) is a failed fit
   and is treated as not resolved. It is excluded from fits and replicate checks, and it
   still gives an MSD bound. On batch 1+2 data the rule excludes six runs: Li10Zn7(PS4)8
   450 K seed 1; Li2GePbS4 800 K seed 0; and seed 0 at 1000 K of Rb2LiTaS4, Li7Y7ZrS16,
   Li2CdGeS4 and Cs2LiNbS4. It changes one class: Li2GePbS4 from SLOW_ESTIMATED to
   SLOW_BOUNDED (σ300 ≤ 0.0072 mS/cm, slow either way). Li10Zn7(PS4)8 goes to the
   terminal rule with or without it. The output keeps the class without this rule in
   `class_without_kinisi_failure_rule`.
3. **Batch 3 (terminal rule).** The four compositions that fail the replicate rule
   receive seeds 3 and 4 at their three eligible temperatures, with the existing run
   lengths:
   - Li5P(S2Cl)2 (600 K seed 2 framework MSD 1.11 Å² > 1, so only 2 usable seeds at
     600 K): 450 and 525 K at 400 ps, 600 K at 200 ps;
   - Li10Zn7(PS4)8 (2 usable seeds at 450 K after item 2; seed 2 and pooled R² < 0.95):
     the same set;
   - Li48P16S61 (pooled R² 0.913): 400, 425 and 450 K at 400 ps;
   - Li6AsS5I (per-seed R² 0.30–0.87, pooled 0.55): 600 K at 200 ps, 700 K at 800 ps,
     800 K at 200 ps.

   That is 24 runs. With five seeds at every fitted temperature the terminal limit is
   reached, so the pooled 95 % interval decides each composition, and none can remain
   without a class.
4. **Reported caveat, no compute.** Li2SiSnS4 is SLOW_BOUNDED at D₀,min = 10⁻⁴ cm²/s
   (σ300 ≤ 0.082 mS/cm) but not at 10⁻⁵ (≤ 0.55). The 800 ps run of A2.3 made 600 K
   eligible, but the slope bound against 525 K is looser than the prefactor bound. The
   verdict stands under the pre-registered D₀,min and is reported as prefactor-dependent.
