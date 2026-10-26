"""The MD runner and batch manifests match what produced the trajectories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import make_campaign

from ssemd import paths

RUNNER = paths.REPO / "production" / "md_runner"
MANIFESTS = paths.REPO / "production" / "manifests"
# sha256 of the bytes of md.py after its module docstring, in the file that ran
# (whole-file sha256 a773248ae302b8fec18c15cbf5caf78c3e7d21efc687c74ce94576eecc8aa0be).
MD_CODE_SHA256 = "87d15adc817c7c4709371234c5d05c2661a9010c15681795611a2be7b3a76d9a"
INITIALIZATION_GUARD_SHA256 = "43ae4b13d1327421c69e882c83bb6aca3a4d7b22b3aaf32e2d404bfcf5de3f35"
MODEL_SHA256 = "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"
TASKS = {"census_12": 36, "census_50": 150, "replicate_seed12": 36, "def_b1": 29, "def_b2": 45, "def_b3": 24}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_md_code_is_the_code_that_ran():
    text = (RUNNER / "md.py").read_text(encoding="utf-8")
    end = text.index('"""', 3) + 3
    assert hashlib.sha256(text[end:].encode()).hexdigest() == MD_CODE_SHA256


def test_initialization_guard_is_unchanged():
    assert _sha(RUNNER / "initialization_guard.py") == INITIALIZATION_GUARD_SHA256


def test_manifests_cover_every_trajectory():
    trajectories = {row.split(",")[7] for row in (paths.PROVENANCE / "trajectories.csv").read_text().splitlines()[1:]}
    seen = set()
    for batch, count in TASKS.items():
        manifest = json.loads((MANIFESTS / f"{batch}.json").read_text(encoding="utf-8"))
        assert len(manifest["tasks"]) == count
        assert manifest["protocol"]["model_sha256"] == MODEL_SHA256
        for task in manifest["tasks"]:
            seen.add(task["trajectory_sha256"])
    assert seen == trajectories


def test_definitive_protocol_hashes_rebuild_exactly():
    for batch in ("def_b1", "def_b2", "def_b3"):
        manifest = json.loads((MANIFESTS / f"{batch}.json").read_text(encoding="utf-8"))
        task_map, _ = make_campaign.runner_tasks(manifest)
        assert [t["protocol_sha256"] for t in task_map] == [t["protocol_sha256"] for t in manifest["tasks"]]


def test_campaign_folder_is_complete(tmp_path):
    out = make_campaign.build(MANIFESTS / "def_b3.json", tmp_path / "campaign")
    for name in ("run_one.py", "task.sh", "pool.pbs", "task_map.json", "manifest.json", "code/src/md.py"):
        assert (out / name).is_file(), name
    assert len(list((out / "structures").glob("*.cif"))) == 4
