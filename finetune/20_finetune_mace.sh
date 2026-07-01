#!/bin/bash
# W7 step 4 — fine-tune MACE-MP-0 (small) on the DFT-labelled Li6PS5Cl snapshots.
# Run INSIDE the Vanda pytorch_2.5 container with PYTHONUSERBASE=~/macepkg (mace_run_train
# ships with mace-torch). Submit as a GPU batch job (A40), walltime ~3 h.
#
# PREREQ: finetune/data/train.xyz + valid.xyz with DFT energy/forces (from steps 2–3).
# The --foundation_model flag turns this into FINE-TUNING (not from scratch): MACE-MP small
# is loaded and adapted to the sulfide data with a small LR.
set -e
export PYTHONUSERBASE="$HOME/macepkg"
export PYTHONPATH="$HOME/macepkg/lib/python3.10/site-packages:$PYTHONPATH"
cd "$HOME/AI4SSB/project2_mlip_md/finetune"

python3 -m mace.cli.run_train \
    --name="li6ps5cl_ft" \
    --foundation_model="small" \
    --train_file="data/train.xyz" \
    --valid_file="data/valid.xyz" \
    --energy_key="dft_energy" \
    --forces_key="dft_forces" \
    --loss="weighted" \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --lr=0.0001 \
    --batch_size=4 \
    --max_num_epochs=120 \
    --swa \
    --default_dtype="float32" \
    --device="cuda" \
    --seed=0 \
    --model_dir="models" \
    --checkpoints_dir="checkpoints" \
    --results_dir="results"

echo "[ft] done -> models/li6ps5cl_ft.model"
echo "[ft] re-run MD with the fine-tuned model, e.g.:"
echo "     python3 02_baseline_md.py --mlip mace --mace-model models/li6ps5cl_ft.model \\"
echo "             --device cuda --temps 600,800,1000 --steps 50000 --traj-tag _ft"
