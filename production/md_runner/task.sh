#!/bin/bash
# Run one task of the task map on this node:   task.sh <index>
# TASK_MAP selects the map (default task_map.json; SMOKE=1 -> smoke_task_map.json).
# Exit 3 = task claimed by another job/worker (not an error).
#
# Ownership is an atomic `mkdir claims/<map>/<index>`, so any number of pools and
# array subjobs can draw from one map without running a task twice. The owner
# line is "<PBS_JOBID> <gpu> <RUN_ID> <utc> <host>". A claim left by an earlier
# run of the *same* PBS job (a PBS rerun, `#PBS -r y`) is taken over once, via a
# second atomic mkdir; claims of other jobs are never touched. On success the
# claim gets a `done` marker (analyze tasks: the claim is released instead); on
# failure it stays, so nobody retries silently (inspect, then
# `rm -r claims/<map>/<index>` to allow a rerun).
#
# Environment by task type:
#   md        container + macepkg (mace-torch 0.3.16 / torch 2.5.0), --nv
#   envcheck, analyze
#             container + analysispkg_py311 (standalone CPython 3.11.15 with the
#             exact requirements-analysis.lock.txt pins), CPU only, 1 thread per
#             process (analyze parallelises with --jobs)
set -euo pipefail
IDX=${1:?Missing task index}
CAMPAIGN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$CAMPAIGN"
# Site paths (container, package tarballs, model, bind mount): site.env.example
source "$CAMPAIGN/site.env"
RUNTIME=$SITE_BASE/runtime
TASK_MAP=${TASK_MAP:-task_map.json}; [[ "${SMOKE:-0}" == 1 ]] && TASK_MAP=smoke_task_map.json
CLAIMS="claims/${TASK_MAP%.json}"
JOB=${PBS_JOBID:?Not in a PBS job}
RUN_ID=${RUN_ID:-$(date +%s)-$$}
GPU=${CUDA_VISIBLE_DEVICES:-none}
mkdir -p "$CLAIMS"
if mkdir "$CLAIMS/$IDX" 2>/dev/null; then
  echo "$JOB $GPU $RUN_ID $(date -u +%FT%TZ) $(hostname -s)" > "$CLAIMS/$IDX/owner"
else
  read -r OJOB _ ORUN _ 2>/dev/null < "$CLAIMS/$IDX/owner" || OJOB=""
  if [[ "$OJOB" == "$JOB" && -n "${ORUN:-}" && "$ORUN" != "$RUN_ID" && ! -e "$CLAIMS/$IDX/done" ]] \
     && mkdir "$CLAIMS/$IDX/rerun_$RUN_ID" 2>/dev/null; then
    echo "Task $IDX: taking over own claim from an earlier run of $JOB (PBS rerun)"
    echo "$JOB $GPU $RUN_ID $(date -u +%FT%TZ) $(hostname -s)" > "$CLAIMS/$IDX/owner"
  else
    echo "Task $IDX: already claimed ($(cat "$CLAIMS/$IDX/owner" 2>/dev/null || echo unknown)); skipping"
    exit 3
  fi
fi

TYPE=$(python3 -c 'import json,sys; i=int(sys.argv[2]); t=json.load(open(sys.argv[1]))[i]; assert t["index"]==i; print(t["task_type"])' "$TASK_MAP" "$IDX")
echo "host=$(hostname -s) job=$JOB map=$TASK_MAP index=$IDX type=$TYPE gpu=$GPU run=$RUN_ID"
export PATH=$SINGULARITY_BIN:$SQUASHFUSE_HOME/bin:$PATH
export LD_LIBRARY_PATH=$SQUASHFUSE_HOME/lib:${LD_LIBRARY_PATH:-}
SIF=$RUNTIME/pytorch_2.5_cuda_12.4.sif
# Verify the cached runtime before extracting/importing it.
echo "d7ec548f406de53d30078cb6caba2079afa1b2d02f597bf27b0edf6a45799dde  $SIF" | sha256sum -c -
LOCAL_RUNTIME=$(mktemp -d /tmp/p2def.XXXXXXXX)
trap 'rm -rf -- "$LOCAL_RUNTIME"' EXIT
export SINGULARITYENV_TASK_MAP="$TASK_MAP" SINGULARITYENV_PBS_JOBID="$JOB" SINGULARITYENV_SMOKE="${SMOKE:-0}"
[[ "$TYPE" == md ]] && THREADS=${OMP_THREADS:-4} || THREADS=1
export SINGULARITYENV_OMP_NUM_THREADS=$THREADS SINGULARITYENV_MKL_NUM_THREADS=$THREADS SINGULARITYENV_OPENBLAS_NUM_THREADS=$THREADS
export SINGULARITYENV_MPLCONFIGDIR="$LOCAL_RUNTIME/mpl-cache"
case "$TYPE" in
  md)
    echo "ebe6dde5a2b885ba8c1702adbd0d05cd1fedbd093766e4ba561e0cb3abd80ba7  $RUNTIME/macepkg.tar" | sha256sum -c -
    tar -xf "$RUNTIME/macepkg.tar" -C "$LOCAL_RUNTIME"
    export SINGULARITYENV_PYTHONUSERBASE="$LOCAL_RUNTIME/macepkg"
    export SINGULARITYENV_PYTHONPATH="$LOCAL_RUNTIME/macepkg/lib/python3.10/site-packages"
    export SINGULARITYENV_P2_MODEL_PATH="$MODEL_PATH"
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    CMD=(singularity exec --nv --bind "$SITE_BIND" --bind "$LOCAL_RUNTIME:$LOCAL_RUNTIME" "$SIF" python3 -u -B "$CAMPAIGN/run_one.py" --index "$IDX")
    ;;
  envcheck|analyze)
    echo "73a89171a5438c306275d419d54e4a28e41b48a6d6775a6cd9d1045251365560  $RUNTIME/analysispkg_py311.tar" | sha256sum -c -
    tar -xf "$RUNTIME/analysispkg_py311.tar" -C "$LOCAL_RUNTIME"
    export SINGULARITYENV_PYTHONNOUSERSITE=1
    CMD=(singularity exec --bind "$SITE_BIND" --bind "$LOCAL_RUNTIME:$LOCAL_RUNTIME" "$SIF" "$LOCAL_RUNTIME/analysispkg/python/bin/python3" -E -s -u -B "$CAMPAIGN/run_one.py" --index "$IDX")
    ;;
  *) echo "Unknown task type $TYPE" >&2; exit 2 ;;
esac
rc=0
"${CMD[@]}" || rc=$?
if (( rc == 0 )); then
  # analyze tasks are re-runnable (the analysis reuses finished runs), so their
  # claim is released; every other task keeps a permanent done marker.
  if [[ "$TYPE" == analyze ]]; then rm -rf -- "${CLAIMS:?}/$IDX"; else touch "$CLAIMS/$IDX/done"; fi
fi
exit $rc
