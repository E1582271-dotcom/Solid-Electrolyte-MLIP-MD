#!/bin/bash
# W7 step 2 prep — install ase into an isolated PYTHONUSERBASE on the VANDA LOGIN node, so the
# labelling driver (11_label_qe.py, pure-python + ase) runs alongside the QE module on Vanda CPU.
# Mirrors setup_mattersim.sh. Run on the Vanda login node (has network); label_qe.pbs reuses ~/asepkg.
set -e
export PYTHONUSERBASE="$HOME/asepkg"
python3 -m pip install --user --no-warn-script-location ase
echo "[setup] ase installed under $PYTHONUSERBASE"
python3 -c "import ase, ase.io.espresso; print('[setup] ase', ase.__version__, '+ io.espresso OK')"
echo "[setup] label_qe.pbs exports PYTHONUSERBASE=$PYTHONUSERBASE automatically."
