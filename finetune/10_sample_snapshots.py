"""
W7 step 1 — sample DECORRELATED snapshots from the baseline MACE-MD trajectories to be
DFT-labelled (Quantum ESPRESSO) and used as a MACE-MP fine-tuning set.

Why decorrelated: consecutive MD frames are highly correlated (the cell barely changes in
1 frame), so labelling all of them wastes DFT time and biases the train set. We take every
`--stride`-th production frame per temperature, pooled across 600/800/1000 K so the training
distribution spans the thermal range the model must get right.

Output: an extxyz with positions + cell ONLY (no energies/forces yet — those are step 2,
filled by DFT). CPU-only, uses ase. Run from the project root or finetune/.

Usage:
    python 10_sample_snapshots.py --stride 100         # ~10 configs/temp -> ~30 total
    python 10_sample_snapshots.py --stride 40 --skip-first 0.2   # denser, drop first 20%
"""
from __future__ import annotations

import argparse
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj-glob", default=os.path.join(PROJ, "data", "traj", "mace_*K.traj"),
                    help="trajectories to sample (default: 50ps baseline MACE runs)")
    ap.add_argument("--stride", type=int, default=100, help="keep every Nth production frame")
    ap.add_argument("--skip-first", type=float, default=0.1,
                    help="drop this leading fraction of each traj (residual equilibration)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "snapshots.xyz"))
    args = ap.parse_args()

    from ase.io import read, write

    traj_files = sorted(glob.glob(args.traj_glob))
    if not traj_files:
        raise SystemExit(f"no trajectories matched {args.traj_glob} -- run 02_baseline_md.py first")

    pooled = []
    for tf in traj_files:
        frames = read(tf, index=":")
        n = len(frames)
        start = int(args.skip_first * n)
        picked = frames[start::args.stride]
        tag = os.path.basename(tf).replace(".traj", "")
        for i, at in enumerate(picked):
            at.info["source"] = tag
            at.info["frame_in_traj"] = start + i * args.stride
            at.info["config_type"] = "Li6PS5Cl_md_snapshot"
        pooled.extend(picked)
        print(f"[10] {tag}: {n} frames -> {len(picked)} sampled (stride {args.stride}, skip {args.skip_first})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write(args.out, pooled, format="extxyz")
    print(f"\n[10] wrote {len(pooled)} snapshots -> {os.path.relpath(args.out, PROJ)}")
    print("[10] next: DFT single-points (step 2, qe_scf_template.in) to add energy+forces, "
          "then fine-tune (20_finetune_mace.sh).")


if __name__ == "__main__":
    main()
