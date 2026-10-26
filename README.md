# Solid-Electrolyte-MLIP-MD

**Which of 62 screened Li–S compositions conduct lithium fast? Molecular dynamics with a universal machine-learning potential, classified under a pre-registered protocol.**

Project 1 of this portfolio screened all 21,712 Li-containing Materials Project entries down to 62 candidate solid-electrolyte compositions. Here every composition is simulated with the universal interatomic potential MACE-MP-0 (320 trajectories, 400–1000 K, up to 800 ps and five MD seeds) and placed in exactly one class by rules fixed before the decisive simulations were run.

![Workflow and outcome](figures/main/fig1_workflow.png)

## Result

All 62 compositions are classified: **11 fast conductors**, 2 slow (measured), 6 slow (two-point estimate) and 43 slow (upper bound); none is thermally unstable down to 400 K.

<!-- BEGIN GENERATED fast-conductors (python -m ssemd docs) -->
| Composition | σ(300 K) (mS cm<sup>−1</sup>) | E<sub>a</sub> (eV) | Decided by |
|---|---:|---:|---|
| Li<sub>7</sub>PS<sub>6</sub> | 85 | 0.174 | six-lead replicate decision |
| Li<sub>5</sub>B<sub>7</sub>S<sub>13</sub> | 38 | 0.176 | six-lead replicate decision |
| Li<sub>4</sub>Re<sub>6</sub>S<sub>11</sub> | 36 | 0.152 | six-lead replicate decision (seeds 0, 1) |
| Li<sub>20</sub>Si<sub>3</sub>P<sub>3</sub>S<sub>23</sub>Cl | 24 (15–37) | 0.202 | replicate rule |
| Li<sub>48</sub>P<sub>16</sub>S<sub>61</sub> | 17 (10–29) | 0.270 | terminal rule |
| Li<sub>10</sub>Zn<sub>7</sub>(PS<sub>4</sub>)<sub>8</sub> † | 10 (3.9–27) | 0.207 | terminal rule |
| Li<sub>9</sub>Nd<sub>2</sub>(PS<sub>4</sub>)<sub>5</sub> | 2.4 (1.1–5.0) | 0.286 | replicate rule |
| Li<sub>3</sub>NbS<sub>4</sub> | 2.0 (0.96–4.2) | 0.352 | replicate rule |
| Li<sub>6</sub>AsS<sub>5</sub>I ‡ | 0.57 (0.042–7.8) | 0.265 | terminal rule |
| Li<sub>5</sub>P(S<sub>2</sub>Cl)<sub>2</sub> † | 0.52 (0.25–1.1) | 0.415 | replicate rule |
| Li<sub>4</sub>TiS<sub>4</sub> | 0.38 | 0.358 | six-lead replicate decision |
<!-- END GENERATED fast-conductors -->

Brackets: pooled 95 % interval of multi-seed fits. † fast only with the failed-fit rule of amendment A3.2 (slow without it). ‡ the interval contains the 0.1 mS cm<sup>−1</sup> threshold. Every composition, with its evidence: [docs/results.md](docs/results.md).

## Approach

1. **Census.** One trajectory per composition at 600, 800 and 1000 K (200 ps, MD seed 0) flagged six leads; seeds 1 and 2 then validated them.
2. **Why a census is not enough.** The census discarded a composition if any temperature failed a quality gate, so "insufficient evidence" mixed compositions in which Li does not move, frameworks that melt only at 800–1000 K in simulation, and 600 K runs too short to resolve diffusion.
3. **Pre-registered classification.** A [protocol](docs/CLASSIFICATION_PROTOCOL.md) committed before the new simulations replaced the veto by per-temperature eligibility (diffusion resolved, framework intact, temperature on target) and defined five classes, a replicate rule, a terminal rule that guarantees a decision, and conservative upper bounds for compositions without resolved diffusion. Three further batches (98 trajectories at 400–800 K, longer runs, up to five seeds) supplied the missing evidence (Fig. 2).
4. **Classes.** Fast conductors need σ(300 K) ≥ 0.1 mS cm<sup>−1</sup> with replicate seeds (Fig. 3); slow classes are measured, estimated from two temperatures, or bounded from above (Fig. 4).

![Arrhenius behaviour of the fast conductors](figures/main/fig3_fast_arrhenius.png)

## Limits

- One un-fine-tuned universal potential at the Materials Project volume, one representative structure per composition, one supercell size. On Li<sub>6</sub>PS<sub>5</sub>Cl the same protocol overpredicts σ(300 K) about 1.8× (separate benchmark project); E<sub>a</sub> and the ordering are more reliable than absolute σ. These are model predictions, not measurements.
- σ(300 K) is a Nernst–Einstein Arrhenius extrapolation from 400–1000 K and neglects ion correlations.
- Amendment A3 was written after the results of the second batch were seen (before the third ran); two fast classes depend on its failed-fit rule. The slow class of Li<sub>2</sub>SiSnS<sub>4</sub> depends on the assumed prefactor (Fig. 5).
- Universal potentials soften the potential-energy surface and can melt frameworks in simulation; melted temperatures are excluded rather than counted against a composition.

## Reproduce

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps
python -m ssemd all       # classification, tables, figures and generated docs from data/
python -m pytest          # 1:1 reproduction of the published classification, data integrity, figures
make check                # all of the above, repository checks, and an unchanged working tree
```

The classification is rebuilt from the committed run records in seconds; the published table is reproduced byte for byte (`tests/test_classification.py`). Raw trajectories are not included ([docs/data.md](docs/data.md)).

## Repository

| Path | Content |
|---|---|
| [docs/results.md](docs/results.md) | every class with its evidence, sensitivity checks, census-to-class reconciliation |
| [docs/methods.md](docs/methods.md) | MD protocol, estimators, eligibility, classification rules, vocabulary |
| [docs/CLASSIFICATION_PROTOCOL.md](docs/CLASSIFICATION_PROTOCOL.md) | the pre-registered protocol with its dated amendments |
| [docs/figures.md](docs/figures.md) | all figures with captions |
| [docs/data.md](docs/data.md), [docs/provenance.md](docs/provenance.md) | data availability, origin of every number, code and model checksums |
| [data/](data/README.md) | run records, compositions, MSD curves, classification, provenance |
| [src/ssemd/](src/ssemd/) | classifier, data loaders, figures, reference estimators |
| [production/](production/README.md) | MD runner and the task manifest of every batch |
| [figures/](figures/) | main and Extended Data figures (PDF, SVG, PNG) with per-panel source data |

## Related projects

Part of a computational portfolio on sulfide solid electrolytes: Project 1 screens the Materials Project and hands over the 62-composition queue; a separate benchmark project calibrates universal potentials against DFT and experiment on Li<sub>6</sub>PS<sub>5</sub>Cl; a Li-metal interface project tests the fast conductors next.

## License

MIT. Author: Jieao Jia.
