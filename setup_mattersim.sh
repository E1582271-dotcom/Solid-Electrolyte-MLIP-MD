#!/bin/bash
# Run INSIDE the Vanda pytorch_2.5 container on the LOGIN node (has internet):
#   module load singularity
#   singularity exec /app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif \
#       bash ~/AI4SSB/project2_mlip_md/setup_mattersim.sh
# Installs MatterSim into an ISOLATED user-site (~/mattersimpkg) so it never clobbers macepkg,
# then pre-downloads the model weights so the offline GPU compute node can run.
set -e
export PYTHONUSERBASE="$HOME/mattersimpkg"
unset PIP_PREFIX
mkdir -p "$HOME/mattersimpkg"

echo "=== START_INSTALL ==="
python3 -m pip install --user --no-cache-dir mattersim 2>&1 | tail -45
echo "=== INSTALL_DONE ==="

export PYTHONPATH="$HOME/mattersimpkg/lib/python3.10/site-packages:$PYTHONPATH"
echo "=== versions ==="
python3 -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| numpy', numpy.__version__)"
python3 -c "import mattersim; print('mattersim', getattr(mattersim, '__version__', '?'))"

echo "=== trigger model download (CPU init) ==="
python3 -c "from mattersim.forcefield import MatterSimCalculator; MatterSimCalculator(device='cpu'); print('MATTERSIM_MODEL_OK')" 2>&1 | tail -12
