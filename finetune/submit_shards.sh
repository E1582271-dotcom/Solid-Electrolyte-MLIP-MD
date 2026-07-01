#!/bin/bash
# W7 — fan out QE Gamma-only DFT labelling of Li6PS5Cl snapshots across CPU shards on Vanda.
# MEASURED: one Gamma SCF is ~24 min on 16 cores (~120 s/iter x ~12 iter, ecutrho 480 -> 135^3 FFT).
# 16 cores is the SWEET SPOT: with npool=1 the parallel-FFT all-to-all saturates ~16 ranks, so 32
# cores is ~3.5x SLOWER PER ITER (anti-scaling) -- do NOT raise NCPUS. Throughput comes from many
# concurrent 1-config jobs (CHUNK=1), which the scheduler load-balances under the per-user cap.
#
# Each shard gets its OWN --workdir and --out so they never race. The config-0 shard (start=0) is
# the canary: as soon as its labelled_s0.xyz gains a frame with dft_energy/dft_forces, the whole
# Gamma pipeline (write_espresso_in kpts=None -> pw.x -> read forces -> extxyz) is validated.
# Concatenate at the end (order is irrelevant for training):
#   cat finetune/data/labelled_s*.xyz > finetune/data/labelled.xyz
# Resumable per shard: re-run to continue a killed shard (skips frames already in its out).
#
# Run from the project root:  bash finetune/submit_shards.sh
set -euo pipefail
NTOT=${NTOT:-27}
START=${START:-0}          # include config 0 (the earlier smoke was walltime-killed, wrote nothing)
CHUNK=${CHUNK:-1}          # 1 config/job -> 27 jobs; scheduler load-balances under the concurrency cap
NCPUS=${NCPUS:-16}         # SWEET SPOT for this Gamma SCF; do NOT raise (npool=1 FFT anti-scales past ~16)
CONVTHR=${CONVTHR:-1e-7}   # on total energy; forces good to <~1 meV/A, ~2-3 iters cheaper than 1e-8
WALL=${WALL:-01:00:00}     # ~24 min/config -> 1 h is ample per single-config job
echo "[shards] NTOT=$NTOT START=$START CHUNK=$CHUNK NCPUS=$NCPUS CONVTHR=$CONVTHR WALL=$WALL"
n=0
for s in $(seq "$START" "$CHUNK" $((NTOT-1))); do
  # paths are relative to finetune/ because label_qe.pbs cd's there before running python
  PW="--start $s --limit $CHUNK --kpts gamma --conv-thr $CONVTHR --workdir qe_work_s$s --out data/labelled_s$s.xyz"
  jid=$(qsub -l walltime="$WALL" \
             -l select=1:ncpus="$NCPUS":mpiprocs="$NCPUS":mem=24gb:switch=cpu_all \
             -N "li6g_s$s" \
             -v "NCPUS=$NCPUS,NPOOL=1,PWARGS=$PW" \
             finetune/label_qe.pbs)
  last=$((s+CHUNK)); [ "$last" -gt "$NTOT" ] && last=$NTOT
  echo "[shards] cfg[$s..$last)  ->  $jid"
  n=$((n+1))
done
echo "[shards] submitted $n shard jobs (chunk $CHUNK, $NCPUS cores each)"
