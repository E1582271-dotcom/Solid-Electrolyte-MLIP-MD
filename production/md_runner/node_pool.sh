#!/bin/bash
# One node of a multi-node GPU pool (launched once per node by pool_multinode.pbs).
#   node_pool.sh <campaign_dir> <task_map> <pbs_jobid> <run_id> <end_epoch> <hosts_csv>
# Everything comes from the arguments: no PBS_* variable (PBS_NODEFILE,
# PBS_VNODENUM, ...) is assumed on this node, and the GPU list is read from this
# node's nvidia-smi (an inherited CUDA_VISIBLE_DEVICES would name the head node's
# GPUs). There is no cross-node barrier: the node runs one worker per local GPU,
# and workers claim tasks through the atomic `mkdir claims/<map>/<index>` on the
# shared file system, exactly as in pool.pbs (longest first, per-task walltime
# fit). All nodes of one job share one RUN_ID, so a node never takes over another
# node's live claim; after a PBS rerun the new RUN_ID retakes this job's claims.
# The node writes node_status/<map>_<job>_node<i>_<host>.txt (rc=0 if it found
# GPUs and no worker failed) and always exits 0; the head job checks the files.
set -uo pipefail
CAMPAIGN=${1:?campaign}; TASK_MAP=${2:?task map}; JOBID=${3:?job id}; RUN_ID=${4:?run id}; END=${5:?end epoch}; HOSTS=${6:?hosts}
cd "$CAMPAIGN" || exit 1
HOST=$(hostname -s)
IDX=-1; n=0
for h in ${HOSTS//,/ }; do [[ "${h%%.*}" == "$HOST" ]] && IDX=$n; n=$(( n + 1 )); done
TAG="${TASK_MAP%.json}_${JOBID%%.*}_node${IDX}_${HOST}"
STATUS="node_status/${TAG}.txt"
mkdir -p logs node_status
exec > "logs/${TAG}.log" 2>&1
unset CUDA_VISIBLE_DEVICES SMOKE
export PBS_JOBID="$JOBID" RUN_ID TASK_MAP
CLAIMS="claims/${TASK_MAP%.json}"
mkdir -p "$CLAIMS"
SAFETY_PCT=125   # tolerate tasks 25 % slower than estimated (as pool.pbs)
MIN_LEFT_S=1200  # runtime sha checks + tar + model load + raw QC + margin
SMI=$(command -v nvidia-smi || echo /usr/bin/nvidia-smi)
GPUS=()
while read -r g; do [[ -n "$g" ]] && GPUS+=("$g"); done < <("$SMI" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
ORDER=(); EST=()
while read -r i s w; do
  [[ "$w" == gpu_pool ]] && { ORDER+=("$i"); EST+=("$s"); }
done < <(python3 -c 'import json,sys; t=json.load(open(sys.argv[1])); [print(x["index"], int(round(x["est_wall_h"]*3600)), x.get("where", "gpu_pool")) for x in sorted(t, key=lambda x: (-x["est_wall_h"], x["index"]))]' "$TASK_MAP")
echo "node host=$HOST index=$IDX job=$JOBID run=$RUN_ID map=$TASK_MAP end_epoch=$END gpus=${GPUS[*]:-none} order=${ORDER[*]}"

worker() {
  local gpu=$1 k i next rc left pending oj orun
  while :; do
    next=""; pending=0; left=$(( END - $(date +%s) ))
    for k in "${!ORDER[@]}"; do
      i=${ORDER[$k]}
      [[ -e "$CLAIMS/$i/done" ]] && continue
      if [[ -e "$CLAIMS/$i" ]]; then
        oj=""; orun=""
        read -r oj _ orun _ 2>/dev/null < "$CLAIMS/$i/owner"
        [[ "$oj" == "$JOBID" && -n "$orun" && "$orun" != "$RUN_ID" ]] || continue
      fi
      pending=1
      (( left > EST[k]*SAFETY_PCT/100 + MIN_LEFT_S )) || continue
      next=$i; break
    done
    if [[ -z "$next" ]]; then
      (( pending )) && echo "[$gpu] stop: remaining tasks do not fit the walltime left (${left}s)" || echo "[$gpu] stop: no unclaimed tasks"
      return 0
    fi
    CUDA_VISIBLE_DEVICES=$gpu bash "$CAMPAIGN/task.sh" "$next" > "logs/${TASK_MAP%.json}_${JOBID%%.*}_task${next}.log" 2>&1; rc=$?
    echo "[$gpu] $(date -u +%FT%TZ) task $next rc=$rc"
    (( rc == 0 || rc == 3 )) || { echo "[$gpu] stop: task $next failed"; return 1; }
  done
}

FAILED=0; PIDS=()
for gpu in "${GPUS[@]}"; do worker "$gpu" & PIDS+=($!); sleep 5; done
for p in "${PIDS[@]}"; do wait "$p" || FAILED=$(( FAILED + 1 )); done
RC=0; (( ${#GPUS[@]} > 0 && FAILED == 0 && IDX >= 0 )) || RC=1
GPU_CSV=$(IFS=,; echo "${GPUS[*]:-}")
echo "host=$HOST index=$IDX n_gpus=${#GPUS[@]} gpus=${GPU_CSV:-none} failed_workers=$FAILED rc=$RC finished=$(date -u +%FT%TZ)" > "$STATUS"
cat "$STATUS"
exit 0
