"""One-off import of the frozen Project 2 results from the legacy working repository.

    python production/migrate_legacy.py --legacy ../project2_mlip_md

The legacy repository (git commit LEGACY_COMMIT, clean working tree) holds the
analysed census, replicate and definitive campaigns. This script reads only
tracked, committed tables (never trajectories or posterior arrays), rebuilds the
classifier's run records with the legacy classifier code itself, and writes the
consolidated tables under ``data/``. Every float is written with ``repr`` so the
tables read back to bit-identical values. ``data/provenance/legacy_files.csv``
maps every written file to the legacy files it came from (path and sha256).

Run once; the outputs are committed and everything else in this repository is
built from them.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

LEGACY_COMMIT = "e2b7416c4af887fb9101aa444836164f689999a9"
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

CENSUS_CAMPAIGNS = {"p1_20260908_large200": "census_12", "p1_remaining50_threeT_ready20260912": "census_50"}
DEFINITIVE_CAMPAIGNS = {"p2_definitive_lowT_20261024": ("def_b1", "compute list + A1"),
                        "p2_definitive_batch2_20261024": ("def_b2", "A2"),
                        "p2_definitive_batch3_20261025": ("def_b3", "A3")}
REPLICATE_DIRS = ("validation_seed012_20261010_v4", "validation_remaining3_seed012_20261016_v1")
FROZEN = ("source_data/campaign/definitive_classification_62.csv",
          "source_data/campaign/definitive_classification_62_summary.json",
          "source_data/campaign/definitive_run_records.csv")
REPLICATE_TABLES = {"lead_decisions_6.csv": "lead_decisions.csv", "seed_fits.csv": "seed_fits.csv",
                    "temperature_summary.csv": "temperature_summary.csv"}

RUN_COLUMNS = (
    "record_id", "trajectory_id", "stage", "md_batch", "estimator", "protocol_section",
    "p1_rank", "formula", "candidate_key", "temperature_K", "md_seed", "production_ps",
    "n_atoms", "n_Li", "cell_volume_A3", "temperature_mean_K", "temperature_bias_fraction",
    "D_cm2_s", "D_lo_cm2_s", "D_hi_cm2_s", "D_direct_cm2_s", "Li_msd_100ps_A2", "D_bound_cm2_s",
    "framework_msd_100ps_A2", "framework_fraction_gt2A_200ps", "diffusion_resolved", "framework_intact",
    "eligible_census_criteria", "kinisi_fit_failed", "eligible", "ne_factor_mS_cm_per_cm2_s",
    "energy_slope_meV_atom_ps", "D_pymatgen_cm2_s", "review_flags",
    "used_by_classifier", "superseded_by", "trajectory_sha256", "legacy_sources",
)
# legacy run-record key -> runs.csv column
RENAMED = {"D_kinisi_cm2_s": "D_cm2_s", "D_kinisi_lo_cm2_s": "D_lo_cm2_s", "D_kinisi_hi_cm2_s": "D_hi_cm2_s",
           "D_direct_MSD_cm2_s": "D_direct_cm2_s", "D_bound_msd_cm2_s": "D_bound_cm2_s"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def cell(value) -> str:
    """Text for one CSV cell: repr for floats (round-trips exactly), lower-case booleans."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([cell(row.get(column)) for column in columns])
    path.write_text(buffer.getvalue(), encoding="utf-8")


def write_csv_gz(path: Path, columns, rows) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([cell(row.get(column)) for column in columns])
    raw = buffer.getvalue().encode("utf-8")
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0, compresslevel=9) as handle:
        handle.write(raw)
    path.write_bytes(out.getvalue())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class Legacy:
    """Read access to the legacy repository, restricted to tracked files at LEGACY_COMMIT."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        head = self._git("rev-parse", "HEAD").strip()
        if head != LEGACY_COMMIT:
            raise SystemExit(f"Legacy HEAD is {head}, expected {LEGACY_COMMIT}")
        if self._git("status", "--porcelain", "--untracked-files=no").strip():
            raise SystemExit("Legacy working tree has uncommitted changes to tracked files")
        self.tracked = set(self._git("ls-files").splitlines())
        self.read_log: dict[str, str] = {}

    def _git(self, *args: str) -> str:
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True,
                              text=True).stdout

    def path(self, relpath: str) -> Path:
        if relpath not in self.tracked:
            raise FileNotFoundError(f"Not a tracked legacy file: {relpath}")
        path = self.root / relpath
        self.read_log[relpath] = sha256_file(path)
        return path

    def csv(self, relpath: str) -> list[dict[str, str]]:
        return read_csv(self.path(relpath))

    def json(self, relpath: str):
        return json.loads(self.path(relpath).read_text(encoding="utf-8"))


def load_legacy_classifier(legacy: Legacy):
    path = legacy.path("src/definitive_classification.py")
    spec = importlib.util.spec_from_file_location("legacy_definitive_classification", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# run records
# ---------------------------------------------------------------------------
def build_runs(legacy: Legacy, dc) -> tuple[list[dict], dict[str, str]]:
    lineage: dict[str, list[str]] = defaultdict(list)
    summary_path = "source_data/campaign/trajectory_summary_186.csv"
    detailed_path = "campaigns/p1_all62_analysis/detailed_trajectory_186.csv"
    summary_rows = legacy.csv(summary_path)
    census = dc.census_runs(summary_rows, legacy.csv(detailed_path))
    by_census_key = {(row["candidate_key"], int(row["temperature_K"])): row for row in summary_rows}
    for directory in REPLICATE_DIRS:
        for name in ("seed012_transport_source.csv", "seed012_primary_MSD_source.csv"):
            legacy.path(f"source_data/campaign/validation_v2/figure_sources/{directory}/{name}")
    replicates = dc.replicate_runs(dc.REPLICATE_SOURCE_DIRS, census)
    definitive = {}
    for folder in DEFINITIVE_CAMPAIGNS:
        rel = f"campaigns/{folder}/analysis/trajectory_results.csv"
        legacy.path(rel)
        definitive[folder] = dc.read_new_runs(legacy.root / rel)
    new_runs = [run for folder in DEFINITIVE_CAMPAIGNS for run in definitive[folder]]
    grouped = dc.assemble_runs(census, replicates, new_runs, replace_existing=True)
    used = {id(run) for runs in grouped.values() for run in runs}
    slot_owner = {(run["candidate_key"], run["temperature_K"], run["md_seed"]): run
                  for runs in grouped.values() for run in runs}

    records: list[dict] = []
    record_of: dict[int, dict] = {}

    def add(run: dict, stage: str, md_batch: str, estimator: str, section: str, record_id: str,
            trajectory_id: str, extra: dict | None = None) -> None:
        row = {RENAMED.get(key, key): value for key, value in run.items() if not key.startswith("_")}
        row.pop("eligible", None)
        row.pop("source_run_relpath", None)
        row.update(record_id=record_id, trajectory_id=trajectory_id, stage=stage, md_batch=md_batch,
                   estimator=estimator, protocol_section=section,
                   temperature_K=int(run["temperature_K"]),
                   eligible_census_criteria=run["eligible"], kinisi_fit_failed=dc.kinisi_fit_failed(run),
                   eligible=bool(run["eligible"] and not dc.kinisi_fit_failed(run)),
                   used_by_classifier=id(run) in used,
                   legacy_sources=";".join(run.get("_sources", ())))
        row.update(extra or {})
        records.append(row)
        record_of[id(run)] = row
        lineage[record_id].append(run.get("source_run_relpath", ""))

    def tag(key: str, temperature: float, seed: int) -> str:
        return f"{key}/{int(temperature)}K_s{seed}"

    census_batch = {}
    for run in census:
        summary = by_census_key[(run["candidate_key"], int(run["temperature_K"]))]
        md_batch = CENSUS_CAMPAIGNS[summary["campaign"]]
        name = tag(run["candidate_key"], run["temperature_K"], run["md_seed"])
        census_batch[(run["candidate_key"], int(run["temperature_K"]))] = md_batch
        add(run, "census", md_batch, "kinisi_mcmc", "census", f"census/{name}", f"{md_batch}/{name}",
            {"energy_slope_meV_atom_ps": dc._float(summary["energy_slope_meV_atom_ps"]),
             "D_pymatgen_cm2_s": dc._float(summary["D_pymatgen_cm2_s"])})
    for run in replicates:
        name = tag(run["candidate_key"], run["temperature_K"], run["md_seed"])
        md_batch = (census_batch[(run["candidate_key"], int(run["temperature_K"]))] if run["md_seed"] == 0
                    else "replicate_seed12")
        add(run, "replicate", md_batch, "fixed_cov_analytic", "replicate validation", f"replicate/{name}",
            f"{md_batch}/{name}")
    for folder, (batch, section) in DEFINITIVE_CAMPAIGNS.items():
        for run in definitive[folder]:
            name = tag(run["candidate_key"], run["temperature_K"], run["md_seed"])
            add(run, "definitive", batch, "kinisi_mcmc", section, f"{batch}/{name}", f"{batch}/{name}")

    for row in record_of.values():
        if not row["used_by_classifier"]:
            owner = slot_owner[(row["candidate_key"], float(row["temperature_K"]), row["md_seed"])]
            row["superseded_by"] = record_of[id(owner)]["record_id"]
    records.sort(key=lambda row: (row["p1_rank"], {"census": 0, "replicate": 1, "definitive": 2}[row["stage"]],
                                  row["md_batch"], row["temperature_K"], row["md_seed"]))
    legacy_relpath = {record_id: paths[0] for record_id, paths in lineage.items()}
    return records, legacy_relpath


# ---------------------------------------------------------------------------
# trajectories (provenance)
# ---------------------------------------------------------------------------
def build_trajectories(legacy: Legacy, records: list[dict]) -> list[dict]:
    census_sha = {}
    large = legacy.json("campaigns/p1_20260908_large200/remote_result_sha256.json")
    census_sha.update({path: digest for path, digest in large.items()})
    rest = legacy.json("campaigns/p1_remaining50_threeT_ready20260912/remote_results_sha256.json")
    census_sha.update({item["path"]: item["sha256"] for item in rest["files"]})
    inventory = {(row["candidate_key"], int(row["temperature_K"]), int(row["md_seed"])): row
                 for row in legacy.csv("source_data/campaign/validation_v2/trajectory_inventory_54.csv")}
    trajectories: dict[str, dict] = {}
    for row in records:
        key, temperature, seed = row["candidate_key"], row["temperature_K"], row["md_seed"]
        if row["md_batch"].startswith("census"):
            digest = census_sha[f"results/{key}/data/traj/mace_{temperature}K.traj"]
            archive = f"{row['md_batch']}/{key}/mace_{temperature}K.traj"
        elif row["md_batch"] == "replicate_seed12":
            digest = inventory[(key, temperature, seed)]["trajectory_sha256"]
            archive = f"replicate_seed12/{key}/{temperature}K_seed{seed}/md.traj"
        else:
            digest = row["trajectory_sha256"]
            archive = f"{row['md_batch']}/{key}/{temperature}K_seed{seed}/md.traj"
        if row["stage"] == "replicate" and seed == 0:
            if inventory[(key, temperature, 0)]["trajectory_sha256"] != digest:
                raise ValueError(f"Replicate seed-0 trajectory differs from the census one: {key} {temperature} K")
        if row["trajectory_sha256"] and row["trajectory_sha256"] != digest:
            raise ValueError(f"Trajectory sha disagrees for {row['record_id']}")
        row["trajectory_sha256"] = digest
        entry = trajectories.setdefault(row["trajectory_id"], {
            "trajectory_id": row["trajectory_id"], "md_batch": row["md_batch"], "candidate_key": key,
            "p1_rank": row["p1_rank"], "temperature_K": temperature, "md_seed": seed,
            "production_ps": row["production_ps"], "trajectory_sha256": digest, "archive_relpath": archive,
            "analysis_records": []})
        if entry["trajectory_sha256"] != digest:
            raise ValueError(f"Two sha256 values for {row['trajectory_id']}")
        entry["analysis_records"].append(row["record_id"])
    rows = sorted(trajectories.values(), key=lambda item: (item["p1_rank"], item["md_batch"], item["temperature_K"],
                                                            item["md_seed"]))
    for item in rows:
        item["analysis_records"] = ";".join(item["analysis_records"])
    return rows


# ---------------------------------------------------------------------------
# MSD curves and window sensitivity
# ---------------------------------------------------------------------------
MSD_COLUMNS = ("record_id", "lag_ps", "Li_x_A2", "Li_y_A2", "Li_z_A2", "Li_total_A2", "framework_A2")


def build_msd(legacy: Legacy, records: list[dict]) -> list[dict]:
    rows = []
    for source in legacy.csv("source_data/campaign/trajectory_source_186.csv"):
        record_id = f"census/{source['candidate_key']}/{int(source['temperature_K'])}K_s{int(source['md_seed'])}"
        rows.append({"record_id": record_id, **{column: source[column] for column in MSD_COLUMNS[1:]}})
    for directory in REPLICATE_DIRS:
        rel = f"source_data/campaign/validation_v2/figure_sources/{directory}/seed012_primary_MSD_source.csv"
        for source in legacy.csv(rel):
            record_id = f"replicate/{source['candidate_key']}/{int(source['temperature_K'])}K_s{int(source['md_seed'])}"
            rows.append({"record_id": record_id, "lag_ps": source["lag_ps"], "Li_total_A2": source["Li_MSD_A2"]})
    for row in records:
        if row["stage"] != "definitive":
            continue
        folder = next(name for name, (batch, _) in DEFINITIVE_CAMPAIGNS.items() if batch == row["md_batch"])
        rel = (f"campaigns/{folder}/analysis/runs/{row['candidate_key']}/"
               f"{row['temperature_K']}K_seed{row['md_seed']}/msd.csv")
        for source in legacy.csv(rel):
            rows.append({"record_id": row["record_id"], **{column: source[column] for column in MSD_COLUMNS[1:]}})
    known = {row["record_id"] for row in records}
    missing = {row["record_id"] for row in rows} - known
    if missing:
        raise ValueError(f"MSD curves for unknown records: {sorted(missing)[:5]}")
    counts = Counter(row["record_id"] for row in rows)
    if len(counts) != len(records):
        raise ValueError(f"MSD curves cover {len(counts)} of {len(records)} records")
    return rows


WINDOW_COLUMNS = ("record_id", "axis", "setting", "start_ps", "end_ps", "cond_max", "D_cm2_s", "D_lo_cm2_s",
                  "D_hi_cm2_s", "D_direct_cm2_s", "covariance_condition", "error")
CENSUS_FRAME_PS = 0.05


def build_windows(legacy: Legacy, records: list[dict]) -> list[dict]:
    rows = []
    for source in legacy.csv("source_data/campaign/window_sensitivity_1302.csv"):
        record_id = f"census/{source['candidate_key']}/{int(source['temperature_K'])}K_s{int(source['md_seed'])}"
        rows.append({"record_id": record_id, "axis": source["analysis_axis"], "setting": source["setting"],
                     "start_ps": repr(int(source["window_start_frame"]) * CENSUS_FRAME_PS),
                     "end_ps": repr(int(source["window_end_frame"]) * CENSUS_FRAME_PS),
                     "cond_max": source["covariance_cond_max"], "D_cm2_s": source["D_cm2_s"],
                     "D_lo_cm2_s": source["D_lo_cm2_s"], "D_hi_cm2_s": source["D_hi_cm2_s"],
                     "D_direct_cm2_s": source["diagnostic_D_cm2_s"],
                     "covariance_condition": source["covariance_condition"], "error": source["error"]})
    for folder, (batch, _) in DEFINITIVE_CAMPAIGNS.items():
        for source in legacy.csv(f"campaigns/{folder}/analysis/window_sensitivity.csv"):
            record_id = f"{batch}/{source['candidate_key']}/{int(source['temperature_K'])}K_s{int(source['md_seed'])}"
            rows.append({"record_id": record_id, "axis": "trajectory_window", "setting": source["window"],
                         "start_ps": source["start_ps"], "end_ps": source["end_ps"], "cond_max": source["cond_max"],
                         "D_cm2_s": source["D_cm2_s"], "D_lo_cm2_s": source["D_lo"], "D_hi_cm2_s": source["D_hi"],
                         "D_direct_cm2_s": source["diagnostic_D_cm2_s"],
                         "covariance_condition": source["covariance_condition"], "error": source["error"]})
    known = {row["record_id"] for row in records}
    unknown = {row["record_id"] for row in rows} - known
    if unknown:
        raise ValueError(f"Window rows for unknown records: {sorted(unknown)[:5]}")
    return rows


# ---------------------------------------------------------------------------
# compositions, structures
# ---------------------------------------------------------------------------
COMPOSITION_COLUMNS = (
    "p1_rank", "formula", "candidate_key", "composition_key", "mp_id",
    "p1_score_log10_S_cm", "p1_evidence_state", "p1_ensemble_spread", "p1_ad_distance",
    "mp_entries_for_composition", "composition_min_e_above_hull_eV_atom", "composition_max_band_gap_eV",
    "spacegroup_no", "crystal_system", "e_above_hull_eV_atom", "band_gap_eV", "mp_theoretical", "icsd_ids",
    "unit_cell_atoms", "supercell_repetitions", "n_atoms", "n_Li", "cell_lengths_A", "cell_angles_deg",
    "cell_volume_A3", "structure_file", "structure_sha256", "census_md_batch",
    "census_screen_label", "census_review_reasons", "census_category", "replicate_outcome",
)


def build_compositions(legacy: Legacy) -> tuple[list[dict], list[tuple[str, Path]]]:
    catalog = legacy.csv("source_data/campaign/sample_catalog_62.csv")
    non_leads = {row["candidate_key"]: row for row in legacy.csv("source_data/campaign/non_lead_classification_56.csv")}
    decisions = {row["candidate_key"]: row for row in legacy.csv("source_data/campaign/validation_v2/lead_decisions_6.csv")}
    manifests = {}
    for folder, batch in CENSUS_CAMPAIGNS.items():
        for candidate in legacy.json(f"campaigns/{folder}/manifest.json")["candidates"]:
            manifests[candidate["key"]] = (folder, batch, candidate)
    rows, structures = [], []
    for source in catalog:
        key = source["candidate_key"]
        folder, batch, candidate = manifests[key]
        structure = legacy.path(f"campaigns/{folder}/{candidate['structure']}")
        if sha256_file(structure) != candidate["structure_sha256"]:
            raise ValueError(f"Structure sha mismatch for {key}")
        structures.append((f"structures/{key}.cif", structure))
        if key in decisions:
            category = "lead"
        else:
            category = non_leads[key]["category"]
        rows.append({
            "p1_rank": int(source["p1_rank"]), "formula": source["formula"], "candidate_key": key,
            "composition_key": source["composition_key"], "mp_id": source["representative_mp_id"],
            "p1_score_log10_S_cm": source["p1_score_log10_S_cm"], "p1_evidence_state": source["p1_evidence_state"],
            "p1_ensemble_spread": source["p1_ensemble_spread"], "p1_ad_distance": source["p1_ad_distance"],
            "mp_entries_for_composition": source["composition_mp_entry_count"],
            "composition_min_e_above_hull_eV_atom": source["composition_min_e_above_hull_eV_atom"],
            "composition_max_band_gap_eV": source["composition_max_band_gap_eV"],
            "spacegroup_no": source["representative_spacegroup_no"],
            "crystal_system": source["representative_crystal_system"],
            "e_above_hull_eV_atom": source["representative_e_above_hull_eV_atom"],
            "band_gap_eV": source["representative_band_gap_eV"], "mp_theoretical": source["representative_theoretical"],
            "icsd_ids": source["representative_icsd_ids"], "unit_cell_atoms": source["unit_cell_atoms"],
            "supercell_repetitions": source["supercell_repetitions"], "n_atoms": candidate["n_atoms"],
            "n_Li": candidate["n_Li"], "cell_lengths_A": source["cell_lengths_A"],
            "cell_angles_deg": source["cell_angles_deg"], "cell_volume_A3": source["cell_volume_A3"],
            "structure_file": f"structures/{key}.cif", "structure_sha256": candidate["structure_sha256"],
            "census_md_batch": batch, "census_screen_label": source["p2_review_status"],
            "census_review_reasons": source["p2_review_reasons"], "census_category": category,
            "replicate_outcome": decisions[key]["decision"] if key in decisions else "",
        })
    if [row["p1_rank"] for row in rows] != list(range(1, 63)):
        raise ValueError("Catalogue is not P1 ranks 1-62 in order")
    return rows, structures


# ---------------------------------------------------------------------------
# batch manifests (production/manifests/)
# ---------------------------------------------------------------------------
MANIFESTS = REPO / "production" / "manifests"
PROTOCOL_KEYS = ("mlip", "model_sha256", "md_code_sha256", "dtype", "timestep_fs", "friction_ase_inverse_time",
                 "log_every_steps", "equilibration_steps", "seed_initialization", "volume")
CENSUS_PROTOCOL_NOTE = ("census runner (one trajectory per composition and temperature, seed 0); "
                        "protocol values from the census manifest")


def build_manifests(legacy: Legacy, trajectories: list[dict]) -> list[str]:
    by_traj = {(row["md_batch"], row["candidate_key"], int(row["temperature_K"]), int(row["md_seed"])): row
               for row in trajectories}
    b3 = legacy.json("campaigns/p2_definitive_batch3_20261025/manifest.json")
    invariants = b3["invariants"]
    protocol = {key: invariants[key] for key in PROTOCOL_KEYS}
    protocol["ensemble"] = "NVT Langevin at the MP/PBE cell volume (no relaxation)"
    runtime = invariants["runtime"]
    written = []

    def task(batch, key, formula, temperature, seed, production_steps, equilibration, extra=None):
        traj = by_traj[(batch, key, temperature, seed)]
        entry = {"task_id": f"{key}/{temperature}K_seed{seed}", "candidate_key": key, "p1_formula": formula,
                 "temperature_K": temperature, "md_seed": seed, "production_steps": production_steps,
                 "equilibration_steps": equilibration, "trajectory_sha256": traj["trajectory_sha256"]}
        entry.update(extra or {})
        return entry

    for folder, batch in CENSUS_CAMPAIGNS.items():
        manifest = legacy.json(f"campaigns/{folder}/manifest.json")
        census = manifest["protocol"]
        tasks = [task(batch, c["key"], c["p1_formula"], temperature, census["seed"], census["production_steps"],
                      census["equilibration_steps"])
                 for c in manifest["candidates"] for temperature in census["temperatures_K"]]
        written.append(write_manifest(batch, {
            "batch": batch, "stage": "census", "protocol_section": "census",
            "description": f"seed-0 census, {len(manifest['candidates'])} compositions x 600/800/1000 K x 200 ps",
            "reconstructed": True, "note": CENSUS_PROTOCOL_NOTE,
            "protocol": {**protocol, "md_code_sha256": manifest["p2_code_snapshot_sha256"]["src/md.py"],
                         "supercell_policy": census["supercell_policy"]},
            "tasks": tasks}))

    inventory = legacy.csv("source_data/campaign/validation_v2/trajectory_inventory_54.csv")
    tasks = [task("replicate_seed12", row["candidate_key"], row["formula"], int(row["temperature_K"]),
                  int(row["md_seed"]), 200000, 10000)
             for row in sorted(inventory, key=lambda r: (int(r["p1_rank"]), int(r["md_seed"]), int(r["temperature_K"])))
             if int(row["md_seed"]) in (1, 2)]
    written.append(write_manifest("replicate_seed12", {
        "batch": "replicate_seed12", "stage": "replicate", "protocol_section": "replicate validation",
        "description": "seeds 1 and 2 of the six census leads at 600/800/1000 K x 200 ps (seed 0 is the census run)",
        "reconstructed": True, "note": "census protocol with MD seeds 1 and 2; reconstructed from the trajectory "
                                       "inventory of the replicate analysis",
        "protocol": protocol, "tasks": tasks}))

    for folder, (batch, section) in DEFINITIVE_CAMPAIGNS.items():
        manifest = legacy.json(f"campaigns/{folder}/manifest.json")
        tasks = []
        for item in manifest["tasks"]:
            payload = item["protocol_payload"]
            tasks.append(task(batch, item["candidate_key"], item["p1_formula"], int(item["temperature_K"]),
                              int(item["md_seed"]), int(item["production_steps"]), int(item["equilibration_steps"]),
                              {"group": item["group"], "est_wall_h": item["est_wall_h"],
                               "protocol_sha256": item["protocol_sha256"],
                               "structure_sha256": payload["structure_sha256"],
                               **({"initial_atom_order_and_geometry_sha256":
                                   payload["initial_atom_order_and_geometry_sha256"]}
                                  if "initial_atom_order_and_geometry_sha256" in payload else {})}))
        written.append(write_manifest(batch, {
            "batch": batch, "stage": "definitive", "protocol_section": section,
            "description": manifest["purpose"], "reconstructed": False,
            "protocol": protocol, "runtime": runtime,
            "payload_schema": "p2_definitive_md_protocol_v1",
            "protocol_sha256_definition": manifest["protocol_sha256_definition"], "tasks": tasks}))
    # Canonical initial geometries of the replicate runs continued with seeds 3/4 (batch 1).
    init_dir = REPO / "production" / "initializations"
    init_dir.mkdir(parents=True, exist_ok=True)
    for rel in sorted(path for path in legacy.tracked
                      if path.startswith("campaigns/p2_definitive_lowT_20261024/initializations/")):
        shutil.copyfile(legacy.path(rel), init_dir / Path(rel).name)
        written.append(f"production/initializations/{Path(rel).name}")
    return written


def write_manifest(batch: str, value: dict) -> str:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS / f"{batch}.json"
    path.write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8")
    return path.relative_to(REPO).as_posix()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--legacy", type=Path, required=True, help="path of the legacy project2_mlip_md checkout")
    args = parser.parse_args(argv)
    legacy = Legacy(args.legacy)
    dc = load_legacy_classifier(legacy)
    for rel in ("docs/CLASSIFICATION_PROTOCOL.md", "source_data/campaign/sample_catalog_62.csv",
                "source_data/campaign/non_lead_classification_56.csv",
                "source_data/campaign/validation_v2/lead_decisions_6.csv"):
        legacy.path(rel)

    written: list[tuple[str, list[str], str]] = []  # (new path, legacy sources, transform)

    records, _ = build_runs(legacy, dc)
    trajectories = build_trajectories(legacy, records)
    write_csv(DATA / "runs.csv", RUN_COLUMNS, records)
    written.append(("data/runs.csv", ["source_data/campaign/trajectory_summary_186.csv",
                                      "campaigns/p1_all62_analysis/detailed_trajectory_186.csv",
                                      *(f"source_data/campaign/validation_v2/figure_sources/{d}/seed012_transport_source.csv"
                                        for d in REPLICATE_DIRS),
                                      *(f"source_data/campaign/validation_v2/figure_sources/{d}/seed012_primary_MSD_source.csv"
                                        for d in REPLICATE_DIRS),
                                      *(f"campaigns/{f}/analysis/trajectory_results.csv" for f in DEFINITIVE_CAMPAIGNS)],
                    "merged: legacy classifier run records (census_runs, replicate_runs, read_new_runs)"))

    write_csv(DATA / "provenance" / "trajectories.csv",
              ("trajectory_id", "md_batch", "p1_rank", "candidate_key", "temperature_K", "md_seed", "production_ps",
               "trajectory_sha256", "archive_relpath", "analysis_records"), trajectories)
    written.append(("data/provenance/trajectories.csv",
                    ["campaigns/p1_20260908_large200/remote_result_sha256.json",
                     "campaigns/p1_remaining50_threeT_ready20260912/remote_results_sha256.json",
                     "source_data/campaign/validation_v2/trajectory_inventory_54.csv",
                     *(f"campaigns/{f}/analysis/trajectory_results.csv" for f in DEFINITIVE_CAMPAIGNS)],
                    "merged: one row per unique trajectory"))

    msd = build_msd(legacy, records)
    write_csv_gz(DATA / "msd_curves.csv.gz", MSD_COLUMNS, msd)
    written.append(("data/msd_curves.csv.gz", ["source_data/campaign/trajectory_source_186.csv",
                                               *(f"source_data/campaign/validation_v2/figure_sources/{d}/seed012_primary_MSD_source.csv"
                                                 for d in REPLICATE_DIRS),
                                               *sorted(path for path in legacy.read_log
                                                       if path.startswith("campaigns/p2_definitive_")
                                                       and path.endswith("/msd.csv"))],
                    "merged: text values copied unchanged"))

    windows = build_windows(legacy, records)
    write_csv(DATA / "window_sensitivity.csv", WINDOW_COLUMNS, windows)
    written.append(("data/window_sensitivity.csv", ["source_data/campaign/window_sensitivity_1302.csv",
                                                    *(f"campaigns/{f}/analysis/window_sensitivity.csv"
                                                      for f in DEFINITIVE_CAMPAIGNS)],
                    "merged: census frames converted to ps (50 fs per frame)"))

    compositions, structures = build_compositions(legacy)
    write_csv(DATA / "compositions.csv", COMPOSITION_COLUMNS, compositions)
    written.append(("data/compositions.csv", ["source_data/campaign/sample_catalog_62.csv",
                                              "source_data/campaign/non_lead_classification_56.csv",
                                              "source_data/campaign/validation_v2/lead_decisions_6.csv",
                                              *(f"campaigns/{f}/manifest.json" for f in CENSUS_CAMPAIGNS)],
                    "merged: P1 identity, MD cell and census evidence columns"))
    (DATA / "structures").mkdir(parents=True, exist_ok=True)
    for new, source in structures:
        shutil.copyfile(source, DATA / new)
        old = source.relative_to(legacy.root).as_posix()
        written.append((f"data/{new}", [old], "verbatim"))

    (DATA / "replicates").mkdir(parents=True, exist_ok=True)
    for old_name, new_name in REPLICATE_TABLES.items():
        old = f"source_data/campaign/validation_v2/{old_name}"
        shutil.copyfile(legacy.path(old), DATA / "replicates" / new_name)
        written.append((f"data/replicates/{new_name}", [old], "verbatim"))

    (DATA / "frozen").mkdir(parents=True, exist_ok=True)
    sums = []
    for old in FROZEN:
        target = DATA / "frozen" / Path(old).name
        shutil.copyfile(legacy.path(old), target)
        sums.append(f"{sha256_file(target)}  {target.name}")
        written.append((f"data/frozen/{target.name}", [old], "verbatim"))
    (DATA / "frozen" / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    # Settings hashes of the definitive-campaign analyses (checked against ssemd.transport.kinisi_census).
    settings = {}
    for folder, (batch, _) in DEFINITIVE_CAMPAIGNS.items():
        run = legacy.json(f"campaigns/{folder}/analysis/run.json")
        settings[batch] = {"config_sha256": run["config_sha256"], "code_sha256": run["code_sha256"],
                           "environment": run["environment"], "runs_analysed": len(run["runs"]),
                           "errors": run["errors_present"]}
    (DATA / "provenance" / "analysis_settings.json").write_text(json.dumps(settings, indent=1, sort_keys=True) + "\n",
                                                                encoding="utf-8")
    written.append(("data/provenance/analysis_settings.json",
                    [f"campaigns/{f}/analysis/run.json" for f in DEFINITIVE_CAMPAIGNS],
                    "extracted: analysis code/settings hashes and environment of each definitive campaign"))

    # Li6PS5Cl calibration (the separate MLIP benchmark project), cited as three reference values.
    reference_rows = []
    for rel, model, key in (("data/metrics_prod.json", "MACE-MP-0 small (un-fine-tuned)", "mace"),
                            ("data/metrics_prod.json", "MatterSim (un-fine-tuned)", "mattersim"),
                            ("data/metrics_ft.json", "MACE-MP-0 fine-tuned on DFT (PBE) labels", "mace")):
        metrics = legacy.json(rel)
        fit = metrics["arrhenius"][key]
        experiment = metrics["experiment_mS_cm"]["Li6PS5Cl (sintered)"]
        reference_rows.append({"system": "Li6PS5Cl", "model": model, "sigma300_mS_cm": fit["sigma300_mS_cm"],
                               "sigma300_lo_mS_cm": fit["sigma300_min_mS_cm"],
                               "sigma300_hi_mS_cm": fit["sigma300_max_mS_cm"], "Ea_eV": fit["Ea_eV"],
                               "Ea_err_eV": fit["Ea_err_eV"], "experiment_sintered_mS_cm": experiment,
                               "experiment_mechanochemical_mS_cm": metrics["experiment_mS_cm"]["Li6PS5Cl (mechanochem.)"],
                               "ratio_to_experiment": fit["sigma300_mS_cm"] / experiment})
    write_csv(DATA / "references" / "li6ps5cl_calibration.csv",
              ("system", "model", "sigma300_mS_cm", "sigma300_lo_mS_cm", "sigma300_hi_mS_cm", "Ea_eV", "Ea_err_eV",
               "experiment_sintered_mS_cm", "experiment_mechanochemical_mS_cm", "ratio_to_experiment"), reference_rows)
    written.append(("data/references/li6ps5cl_calibration.csv", ["data/metrics_prod.json", "data/metrics_ft.json"],
                    "extracted: Arrhenius sigma300 and Ea of the Li6PS5Cl benchmark (P6)"))

    (REPO / "docs").mkdir(exist_ok=True)
    shutil.copyfile(legacy.path("docs/CLASSIFICATION_PROTOCOL.md"), REPO / "docs" / "CLASSIFICATION_PROTOCOL.md")
    written.append(("docs/CLASSIFICATION_PROTOCOL.md", ["docs/CLASSIFICATION_PROTOCOL.md"], "verbatim"))

    for path in build_manifests(legacy, trajectories):
        batch = Path(path).stem
        if path.startswith("production/initializations/"):
            sources = []
        elif batch in CENSUS_CAMPAIGNS.values():
            folder = next(f for f, b in CENSUS_CAMPAIGNS.items() if b == batch)
            sources = [f"campaigns/{folder}/manifest.json"]
        elif batch == "replicate_seed12":
            sources = ["source_data/campaign/validation_v2/trajectory_inventory_54.csv"]
        else:
            folder = next(f for f, (b, _) in DEFINITIVE_CAMPAIGNS.items() if b == batch)
            sources = [f"campaigns/{folder}/manifest.json"]
        if path.startswith("production/initializations/"):
            written.append((path, [f"campaigns/p2_definitive_lowT_20261024/initializations/{Path(path).name}"],
                            "verbatim"))
            continue
        written.append((path, sources, "rebuilt: one manifest schema for every MD batch (no site paths)"))

    legacy_rows = []
    for new, sources, transform in written:
        for source in sources:
            digest = legacy.read_log.get(source, "")
            legacy_rows.append({"new_path": new, "legacy_path": source, "legacy_sha256": digest,
                                "transform": transform})
    # Every file the legacy classifier and this import read, including inputs that have no new counterpart.
    mapped = {row["legacy_path"] for row in legacy_rows}
    for source, digest in sorted(legacy.read_log.items()):
        if source not in mapped:
            legacy_rows.append({"new_path": "", "legacy_path": source, "legacy_sha256": digest,
                                "transform": "read (input of the legacy classifier or an integrity check)"})
    write_csv(DATA / "provenance" / "legacy_files.csv", ("new_path", "legacy_path", "legacy_sha256", "transform"),
              legacy_rows)

    stages = Counter(row["stage"] for row in records)
    print(json.dumps({"records": len(records), "by_stage": stages, "used_by_classifier":
                      sum(row["used_by_classifier"] for row in records), "trajectories": len(trajectories),
                      "msd_rows": len(msd), "window_rows": len(windows), "structures": len(structures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
