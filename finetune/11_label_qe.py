"""
W7 step 2 — DFT-label the sampled snapshots with Quantum ESPRESSO (pw.x SCF), producing
energy + forces (+ stress) for MACE-MP fine-tuning.

    finetune/data/snapshots.xyz  --(this)-->  finetune/data/labelled.xyz

Per config: strip the MACE placeholder labels, write a pw.x SCF input (ASE), run pw.x, parse
the result, and append a frame carrying info['dft_energy'], arrays['dft_forces'] and
info['dft_stress'] -- exactly the keys 20_finetune_mace.sh reads.

Settings mirror qe_scf_template.in (all overridable on the CLI):
  * PBE + SSSP-efficiency pseudopotentials, AUTO-DISCOVERED from --pseudo-dir (pseudo/<El>*.UPF),
    fetched by 00_fetch_pseudos.sh. Cutoffs auto-set from pseudo/sssp.json (max over Li/P/S/Cl)
    when present, else --ecutwfc/--ecutrho.
  * gaussian smearing (safe for the metallic-ish Li sublattice), conv_thr 1e-8, tprnfor+tstress.
  * k-mesh from --kpts. CONVERGE ecutwfc + k-mesh on ONE snapshot before mass labelling:
        python 11_label_qe.py --limit 1 --ecutwfc 50 --kpts 2,2,2   # then 60/70, 3,3,3 ...

Resumable: appends to --out and skips already-labelled configs. A failed SCF is logged to
labelling.log and skipped so one bad config never kills the batch.

pw.x invocation: --pw-cmd or $QE_PW_CMD (default 'pw.x'); the PBS wraps it as 'mpirun -np N pw.x'.
Needs pw.x + ase on PATH (Atlas CPU via label_qe.pbs). CPU-only; do NOT label on the A40.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ELEMENTS = ["Li", "P", "S", "Cl"]
# MACE-origin fields to strip so DFT labels never mix with placeholders
_STALE_INFO = ("energy", "free_energy", "stress", "dft_energy", "dft_stress")
_STALE_ARR = ("forces", "momenta", "energies")


def discover_pseudos(pseudo_dir):
    """Map each element to its UPF in pseudo_dir (element-prefixed, case-insensitive)."""
    if not os.path.isdir(pseudo_dir):
        raise SystemExit(f"[11] no pseudo dir {pseudo_dir} -- run 00_fetch_pseudos.sh first")
    upfs = [f for f in os.listdir(pseudo_dir) if f.lower().endswith(".upf")]
    pp = {}
    for el in ELEMENTS:
        cands = [f for f in upfs
                 if f.split(".")[0].split("_")[0].lower() == el.lower()] or \
                [f for f in upfs if f.lower().startswith(el.lower())]
        if not cands:
            raise SystemExit(f"[11] no UPF for {el} in {pseudo_dir} (have: {sorted(upfs)})")
        pp[el] = sorted(cands, key=len)[0]
    return pp


def cutoffs_from_sssp(pseudo_dir, fb_wfc, fb_rho):
    """Recommended (ecutwfc, ecutrho) = max over Li/P/S/Cl from pseudo/sssp.json, else fallback."""
    path = os.path.join(pseudo_dir, "sssp.json")
    if not os.path.exists(path):
        return fb_wfc, fb_rho, "cli-default"
    try:
        d = json.load(open(path))
        wfc = max(float(d[el].get("cutoff_wfc", d[el].get("cutoff", 0))) for el in ELEMENTS)
        rho = max(float(d[el].get("cutoff_rho", d[el].get("rho_cutoff", 0))) for el in ELEMENTS)
        if wfc <= 0:
            return fb_wfc, fb_rho, "cli-default (sssp.json had no cutoffs)"
        return wfc, (rho or 8 * wfc), "sssp.json"
    except Exception as e:
        print(f"[11] WARN could not read sssp.json ({e}); using CLI cutoffs")
        return fb_wfc, fb_rho, "cli-default"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", default=os.path.join(HERE, "data", "snapshots.xyz"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "labelled.xyz"))
    ap.add_argument("--pseudo-dir", default=os.path.join(HERE, "pseudo"))
    ap.add_argument("--pw-cmd", default=os.environ.get("QE_PW_CMD", "pw.x"),
                    help="pw.x invocation; PBS sets $QE_PW_CMD='mpirun -np N pw.x'")
    ap.add_argument("--ecutwfc", type=float, default=60.0, help="Ry (fallback if no sssp.json)")
    ap.add_argument("--ecutrho", type=float, default=480.0, help="Ry (fallback if no sssp.json)")
    ap.add_argument("--kpts", default="2,2,2", help="Monkhorst-Pack mesh; CONVERGE this")
    ap.add_argument("--degauss", type=float, default=0.01)
    ap.add_argument("--conv-thr", type=float, default=1.0e-8)
    ap.add_argument("--workdir", default=os.path.join(HERE, "qe_work"))
    ap.add_argument("--limit", type=int, default=0, help="label only first N (0=all); for convergence")
    args = ap.parse_args()

    from ase.io import read, write
    from ase.io.espresso import write_espresso_in, read_espresso_out

    frames = read(args.snapshots, index=":")
    if args.limit:
        frames = frames[:args.limit]
    pp = discover_pseudos(args.pseudo_dir)
    ecutwfc, ecutrho, src = cutoffs_from_sssp(args.pseudo_dir, args.ecutwfc, args.ecutrho)
    kpts = tuple(int(x) for x in args.kpts.split(","))
    print(f"[11] {len(frames)} snapshots | pseudos {pp} | ecutwfc={ecutwfc} ecutrho={ecutrho} "
          f"({src}) | kpts={kpts} | pw='{args.pw_cmd}'")

    # resume: labelled.xyz is written in input order, so skip the count already present
    done = 0
    if os.path.exists(args.out):
        done = len(read(args.out, index=":"))
        print(f"[11] resume: {done} already labelled in {os.path.relpath(args.out)}")

    input_data = {
        "control": {"calculation": "scf", "tprnfor": True, "tstress": True, "disk_io": "none",
                    "prefix": "li6ps5cl", "pseudo_dir": os.path.abspath(args.pseudo_dir),
                    "outdir": os.path.abspath(args.workdir)},
        "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing",
                   "smearing": "gaussian", "degauss": args.degauss},
        "electrons": {"conv_thr": args.conv_thr, "mixing_beta": 0.4},
    }

    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logf = open(os.path.join(HERE, "labelling.log"), "a")
    ok = fail = 0
    for i, atoms in enumerate(frames):
        if i < done:
            continue
        atoms.calc = None
        for k in _STALE_INFO:
            atoms.info.pop(k, None)
        for k in _STALE_ARR:
            atoms.arrays.pop(k, None)
        inp = os.path.join(args.workdir, f"cfg{i:04d}.in")
        out = os.path.join(args.workdir, f"cfg{i:04d}.out")
        with open(inp, "w") as fd:
            write_espresso_in(fd, atoms, input_data=input_data, pseudopotentials=pp, kpts=kpts)
        try:
            with open(out, "w") as o:
                subprocess.run(f"{args.pw_cmd} -in {inp}", shell=True, check=True,
                               stdout=o, stderr=subprocess.STDOUT)
            with open(out) as fh:
                res = list(read_espresso_out(fh))[-1]
            E, F = res.get_potential_energy(), res.get_forces()
            atoms.info["dft_energy"] = float(E)
            atoms.arrays["dft_forces"] = F
            try:
                atoms.info["dft_stress"] = " ".join(f"{x:.8e}" for x in res.get_stress())
            except Exception:
                pass
            atoms.info["labelled_by"] = "QE-pw.x-PBE-SSSP"
            write(args.out, atoms, format="extxyz", append=True)
            ok += 1
            print(f"[11] cfg{i:04d} OK  E={E:.4f} eV  |F|max={abs(F).max():.3f} eV/A")
        except Exception as e:
            fail += 1
            logf.write(f"cfg{i:04d} FAILED: {e}\n"); logf.flush()
            print(f"[11] cfg{i:04d} FAILED ({e}) -- logged, continuing")
    logf.close()
    print(f"[11] done: +{ok} labelled, {fail} failed -> {os.path.relpath(args.out)}")
    print("[11] next: python 12_split_train_valid.py   then   bash 20_finetune_mace.sh")


if __name__ == "__main__":
    main()
