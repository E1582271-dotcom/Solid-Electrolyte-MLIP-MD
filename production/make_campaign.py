"""Assemble a runnable MD campaign folder from one batch manifest.

    python production/make_campaign.py production/manifests/def_b3.json /path/to/campaign

The folder gets the runner (production/md_runner/), the MD code, the input supercells of the
batch, ``task_map.json`` and a runner ``manifest.json`` whose per-task protocol payload and
``protocol_sha256`` are rebuilt from the batch manifest. For the definitive batches the
rebuilt hashes equal the ones recorded when the batch ran (tests/test_production.py). Copy
``md_runner/site.env.example`` to ``site.env`` in the folder and edit it before submitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "production" / "md_runner"
RUNNER_FILES = ("run_one.py", "task.sh", "pool.pbs", "run.pbs", "node_pool.sh", "pool_multinode.pbs",
                "initialization_guard.py", "site.env.example")
COMPOSITIONS = REPO / "data" / "compositions.csv"
STRUCTURES = REPO / "data"
INITIALIZATIONS = REPO / "production" / "initializations"
PAYLOAD_SCHEMA = "p2_definitive_md_protocol_v1"
# Wording recorded in the payloads of the definitive batches (part of their protocol_sha256).
PAYLOAD_ENSEMBLE = "NVT Langevin at MP/PBE cell volume (no relaxation), matching the census"


def object_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compositions() -> dict[str, dict[str, str]]:
    with COMPOSITIONS.open(newline="", encoding="utf-8") as handle:
        return {row["candidate_key"]: row for row in csv.DictReader(handle)}


def protocol_payload(task: dict, composition: dict, protocol: dict) -> dict:
    return {
        "schema": PAYLOAD_SCHEMA,
        "mlip": protocol["mlip"],
        "model_sha256": protocol["model_sha256"],
        "dtype": protocol["dtype"],
        "md_code_sha256": protocol["md_code_sha256"],
        "ensemble": PAYLOAD_ENSEMBLE,
        "timestep_fs": protocol["timestep_fs"],
        "friction_ase_inverse_time": protocol["friction_ase_inverse_time"],
        "log_every_steps": protocol["log_every_steps"],
        "equilibration_steps": task["equilibration_steps"],
        "production_steps": task["production_steps"],
        "seed": task["md_seed"],
        "seed_initialization": protocol["seed_initialization"],
        "temperature_K": task["temperature_K"],
        "candidate_key": task["candidate_key"],
        "structure_sha256": composition["structure_sha256"],
        "n_atoms": int(composition["n_atoms"]),
        "n_Li": int(composition["n_Li"]),
        "supercell_repetitions": [int(n) for n in composition["supercell_repetitions"].split("x")],
        **({"initial_atom_order_and_geometry_sha256": task["initial_atom_order_and_geometry_sha256"]}
           if "initial_atom_order_and_geometry_sha256" in task else {}),
    }


def runner_tasks(manifest: dict) -> tuple[list[dict], list[dict]]:
    """Task map entries and runner-manifest entries (task + payload) for one batch."""
    table = compositions()
    task_map, manifest_tasks = [], []
    for index, task in enumerate(manifest["tasks"]):
        payload = protocol_payload(task, table[task["candidate_key"]], manifest["protocol"])
        entry = {"index": index, "task_type": "md", "where": "gpu_pool", "group": task.get("group", manifest["batch"]),
                 "task_id": task["task_id"], "candidate_key": task["candidate_key"], "p1_formula": task["p1_formula"],
                 "temperature_K": task["temperature_K"], "md_seed": task["md_seed"],
                 "production_steps": task["production_steps"], "equilibration_steps": task["equilibration_steps"],
                 "output_root": "results", "protocol_sha256": object_sha(payload),
                 "est_wall_h": task.get("est_wall_h", 8.0)}
        task_map.append(entry)
        manifest_tasks.append({**entry, "protocol_payload": payload})
    return task_map, manifest_tasks


def build(manifest_path: Path, out: Path) -> Path:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    table = compositions()
    out.mkdir(parents=True, exist_ok=False)
    for name in RUNNER_FILES:
        shutil.copyfile(RUNNER / name, out / name)
    (out / "code" / "src").mkdir(parents=True)
    shutil.copyfile(RUNNER / "md.py", out / "code" / "src" / "md.py")
    (out / "structures").mkdir()
    keys = sorted({task["candidate_key"] for task in manifest["tasks"]})
    candidates = []
    for key in keys:
        row = table[key]
        shutil.copyfile(STRUCTURES / row["structure_file"], out / "structures" / f"{key}.cif")
        candidate = {"key": key, "p1_formula": row["formula"], "structure": f"structures/{key}.cif",
                     "structure_sha256": row["structure_sha256"], "n_atoms": int(row["n_atoms"]),
                     "n_Li": int(row["n_Li"])}
        initial = {task.get("initial_atom_order_and_geometry_sha256") for task in manifest["tasks"]
                   if task["candidate_key"] == key} - {None}
        if initial:
            # Continued replicates start from the canonical geometry of the earlier seeds.
            source = INITIALIZATIONS / f"{key}.json"
            (out / "initializations").mkdir(exist_ok=True)
            shutil.copyfile(source, out / "initializations" / source.name)
            candidate.update(canonical_initialization_path=f"initializations/{source.name}",
                             canonical_initialization_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             initial_atom_order_and_geometry_sha256=initial.pop())
        candidates.append(candidate)
    task_map, manifest_tasks = runner_tasks(manifest)
    (out / "task_map.json").write_text(json.dumps(task_map, indent=1) + "\n", encoding="utf-8")
    runner_manifest = {"campaign": out.name, "batch": manifest["batch"], "protocol": manifest["protocol"],
                       "candidates": candidates, "tasks": manifest_tasks, "smoke_tasks": []}
    (out / "manifest.json").write_text(json.dumps(runner_manifest, indent=1) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path, help="new campaign folder (must not exist)")
    args = parser.parse_args(argv)
    print(build(args.manifest, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
