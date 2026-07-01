"""
W7 step 3 — split the DFT-labelled snapshots into train/valid for mace_run_train.

    finetune/data/labelled.xyz  --(this)-->  finetune/data/train.xyz + valid.xyz

Deterministic shuffle (fixed seed) so the split is reproducible. Keys (dft_energy / dft_forces)
are carried through untouched -- 20_finetune_mace.sh reads them via --energy_key/--forces_key.

Usage:
    python 12_split_train_valid.py                 # 10% validation
    python 12_split_train_valid.py --frac-valid 0.15 --seed 1
"""
from __future__ import annotations

import argparse
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", default=os.path.join(HERE, "data", "labelled.xyz"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--frac-valid", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ase.io import read, write

    frames = read(args.labelled, index=":")
    n = len(frames)
    if n < 4:
        raise SystemExit(f"[12] only {n} labelled configs -- label more before splitting")
    missing = [i for i, a in enumerate(frames) if "dft_energy" not in a.info]
    if missing:
        raise SystemExit(f"[12] {len(missing)} configs lack dft_energy (e.g. idx {missing[:5]})")

    idx = list(range(n))
    random.Random(args.seed).shuffle(idx)
    n_val = max(1, int(round(args.frac_valid * n)))
    val_idx = set(idx[:n_val])
    train = [frames[i] for i in range(n) if i not in val_idx]
    valid = [frames[i] for i in idx[:n_val]]

    tp = os.path.join(args.out_dir, "train.xyz")
    vp = os.path.join(args.out_dir, "valid.xyz")
    write(tp, train, format="extxyz")
    write(vp, valid, format="extxyz")
    print(f"[12] {n} labelled -> {len(train)} train + {len(valid)} valid (seed {args.seed})")
    print(f"[12] wrote {os.path.relpath(tp)}, {os.path.relpath(vp)}")
    print("[12] next: bash 20_finetune_mace.sh   (GPU / A40 container)")


if __name__ == "__main__":
    main()
