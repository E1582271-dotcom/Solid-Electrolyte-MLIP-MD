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
| 1. Sample N decorrelated snapshots from trajectories | `10_sample_snapshots.py` (ase) | seconds, CPU | ✅ done → `data/snapshots.xyz` |
| 2. DFT single-points → energy+forces labels | `11_label_qe.py` + `label_qe.pbs` (QE `pw.x`) | ~N × (0.5–2 h) CPU | 🟡 scaffolded — needs pseudos + a QE run |
| 3. Build train/valid extxyz | `12_split_train_valid.py` | seconds | 🟡 scaffolded |
| 4. Fine-tune MACE-MP | `20_finetune_mace.sh` (`mace_run_train --foundation_model`) | ~1–3 h A40 | 🟡 scaffolded |
| 5. Re-run MD with fine-tuned model → re-analyse σ | `02_baseline_md.py --mace-model <path>` + `03` | ~1–2 h A40 | ⏳ |

**N**: a demonstrative fine-tune wants ~100–300 labelled configs (spanning 600/800/1000 K +
the relaxed cell). More configs → better, with diminishing returns. `snapshots.xyz` currently
carries MACE energies/forces as **placeholders** — step 2 overwrites them with DFT (`dft_energy`,
`dft_forces`); you cannot fine-tune MACE on MACE labels.

## Files
- `10_sample_snapshots.py` — decorrelated sampling from `data/traj*/*.traj` → `data/snapshots.xyz`.
- `00_fetch_pseudos.sh` — **login node**: fetch SSSP-efficiency PBE UPFs → `pseudo/` (auto-discovered).
- `setup_qe_env.sh` — **login node**: install `ase` into `~/asepkg` for the driver.
- `11_label_qe.py` — QE SCF per snapshot → `data/labelled.xyz` (resumable; per-config failures logged).
- `label_qe.pbs` — Vanda CPU job wrapping the driver (`mpirun pw.x`, no `-P`, verify `QE_MODULE`).
- `12_split_train_valid.py` — `labelled.xyz` → `train.xyz` + `valid.xyz`.
- `qe_scf_template.in` — reference spec of the SCF namelists (executable version = `11_label_qe.py`).
- `20_finetune_mace.sh` — `mace_run_train` fine-tune (foundation = MACE-MP small; reads `dft_*` keys).

## Concrete run order (W7 step 2→4) — all on **Vanda** (personal free budget, no `-P`)
```bash
# --- Vanda LOGIN node (NUS-id@vanda.nus.edu.sg; compute nodes are offline) ---
bash finetune/00_fetch_pseudos.sh          # or SSSP_TAR_URL=... bash 00_fetch_pseudos.sh
bash finetune/setup_qe_env.sh
module avail 2>&1 | grep -i espresso       # <- note the real QE module name

# --- CONVERGE first (one snapshot; bump ecutwfc 50→60→70, kpts 2,2,2→3,3,3) ---
qsub -v NCPUS=36,QE_MODULE=<name>,PWARGS='--limit 1 --ecutwfc 60 --kpts 2,2,2' finetune/label_qe.pbs

# --- Mass label on Vanda CPU (resumable: re-qsub to continue past the walltime) ---
qsub -v NCPUS=36,QE_MODULE=<name> finetune/label_qe.pbs

# --- Split, then fine-tune on the A40 (container, PYTHONUSERBASE=~/macepkg) ---
python finetune/12_split_train_valid.py
qsub 20_finetune_mace.sh                    # then step 5: re-run 02/03 with models/li6ps5cl_ft.model
```
Everything lives on **Vanda**: DFT labelling on the CPU partition (personal free 10k CPUhr/yr),
fine-tune + MD on the A40 (personal free 1k GPUhr/yr). Pseudopotentials: SSSP **efficiency PBE**
for Li P S Cl (`pseudo/`, gitignored). A 52-atom Γ-ish SCF is minutes–hours on 36 CPU cores.
