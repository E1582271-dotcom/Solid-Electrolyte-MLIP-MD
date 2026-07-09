#!/bin/bash
# Runs INSIDE the Vanda pytorch_2.5_cuda_12.4_unsloth.sif container (called by run_md.pbs).
# MACE 0.3.16 + ase live in ~/macepkg (installed against this container's py3.10/torch2.5).
# Parameters arrive as env vars (set by run_md.pbs / run_leads.pbs via SINGULARITYENV_*): MLIP TEMPS
# STEPS EQUILIB LOG_EVERY TRAJ_TAG SUPERCELL_TAG MACE_MODEL LEAD_CIF (W11 leads: explicit structure).
set -e
# PKGBASE selects the user-site package tree (macepkg for MACE, mattersimpkg for MatterSim;
# isolated so MatterSim's deps never clobber the working MACE install).
PKGBASE="${PKGBASE:-macepkg}"
export PYTHONUSERBASE="$HOME/$PKGBASE"
export PYTHONPATH="$HOME/$PKGBASE/lib/python3.10/site-packages:$PYTHONPATH"
cd "$HOME/AI4SSB/project2_mlip_md"

python3 -c "import torch; print('[gpu]', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"

# TEMPS uses '_' as separator (qsub -v reserves ',') -> convert back to commas
TEMPS_CSV="${TEMPS//_/,}"

python3 02_baseline_md.py \
  --mlip "${MLIP:-mace}" \
  --device cuda \
  --temps "${TEMPS_CSV:-800}" \
  --steps "${STEPS:-2000}" \
  --equilib "${EQUILIB:-500}" \
  --log-every "${LOG_EVERY:-50}" \
  --supercell-tag "${SUPERCELL_TAG:-}" \
  --mace-model "${MACE_MODEL:-small}" \
  --traj-tag "${TRAJ_TAG:-}" \
  ${LEAD_CIF:+--cif "$LEAD_CIF"}
