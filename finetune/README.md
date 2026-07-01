# W7 — Fine-tuning MACE-MP-0 for Li₆PS₅Cl (scaffold)

The baseline (un-fine-tuned MACE-MP-0) over-predicts σ(300 K) by ~9.4× with E_a slightly
low (0.198 eV). W7 asks: **does sulfide-specific fine-tuning pull σ toward experiment?**

## The honest bottleneck: training data
Fine-tuning needs **DFT labels** (energy + forces, ideally stress) on diverse Li₆PS₅Cl
configurations. You cannot fine-tune MACE on MACE labels (circular). So the real work is
*getting DFT-labelled configs*. Decision (self-consistent active learning):

> **Sample decorrelated snapshots from our own MACE-MD trajectories → DFT single-points
> (Quantum ESPRESSO, free/open-source) → fine-tune MACE-MP on them.**

Why this route:
- Uses configs the model *actually visits* during MD (the distribution that matters), not
  random or only-equilibrium structures.
- QE is free (no VASP licence); a 52-atom Γ-ish SCF is minutes–hours on CPU → fits **Atlas
  free CPU** or Vanda CPU. (A40 is poor at FP64/DFT — do labelling on CPU, not the A40.)
- Mirrors how MACE-MP itself was trained, just specialised to the argyrodite.

## Honest scope (this is a multi-day sub-project, NOT a one-run task)
| Step | Tool | Cost | Status |
|---|---|---|---|
| 1. Sample N decorrelated snapshots from trajectories | `10_sample_snapshots.py` (ase) | seconds, CPU | ✅ runnable now |
| 2. DFT single-points → energy+forces labels | Quantum ESPRESSO (`qe_scf_template.in`) | ~N × (0.5–2 h) CPU | ⏳ needs QE + pseudopotentials |
| 3. Build train/valid extxyz with labels | ase / mace tooling | minutes | ⏳ |
| 4. Fine-tune MACE-MP | `20_finetune_mace.sh` (`mace_run_train --foundation_model`) | ~1–3 h A40 | ⏳ |
| 5. Re-run MD with fine-tuned model → re-analyse σ | `02_baseline_md.py --mace-model <path>` + `03` | ~1–2 h A40 | ⏳ |

**N**: a demonstrative fine-tune wants ~100–300 labelled configs (spanning 600/800/1000 K +
the relaxed cell). More configs → better, with diminishing returns.

## Files
- `10_sample_snapshots.py` — decorrelated sampling from `data/traj*/*.traj` → `finetune/data/snapshots.xyz`.
- `qe_scf_template.in` — QE `pw.x` SCF template for the 52-atom cell (fill in pseudopotential
  paths + tune ecutwfc/k-points; convergence-test before mass labelling).
- `20_finetune_mace.sh` — `mace_run_train` fine-tune command template (foundation = MACE-MP small).

## Immediate next concrete step
Run step 1 (done in this session) to get `snapshots.xyz`, then decide labelling compute:
QE on **Atlas free CPU** (recommended) vs request a small Hopper allocation. Pseudopotentials:
SSSP efficiency set (PBEsol or PBE) for Li, P, S, Cl.
