"""Derived tables: the workflow funnel (data/workflow.csv) and the generated docs/results.md."""
from __future__ import annotations

from collections import Counter

from . import data, paths
from .records import write_csv

# Upstream counts from Project 1 (Solid-Electrolyte-Screening): experimental records, the frozen
# Materials Project funnel and the composition queue handed to this project.
UPSTREAM = (
    ("experimental_records", 599, "OBELiX records", "478 official train + 121 held-out test"),
    ("mp_li_entries", 21712, "MP entries", "all Li-containing Materials Project entries (database 2026.04.13)"),
    ("mp_funnel", 248, "MP entries", "S, no H, 3-5 elements, E_hull <= 0.05 eV/atom, band gap >= 1.5 eV"),
    ("candidate_structures", 79, "MP entries", "P1 screening candidates before grouping by composition"),
)
WORKFLOW_COLUMNS = ("stage", "count", "unit", "detail")


def workflow_rows() -> list[dict]:
    runs = data.runs()
    compositions = data.compositions()
    classes = Counter(data.classification()["class"])
    trajectories = runs.drop_duplicates("trajectory_id")
    batches = Counter(trajectories["md_batch"])
    leads = int((compositions["census_category"] == "lead").sum())
    rows = [dict(zip(WORKFLOW_COLUMNS, row)) for row in UPSTREAM]
    rows += [
        {"stage": "queue", "count": len(compositions), "unit": "compositions",
         "detail": "frozen P1 queue; one representative MP structure simulated per composition"},
        {"stage": "census", "count": batches["census_12"] + batches["census_50"], "unit": "trajectories",
         "detail": "62 compositions x 600/800/1000 K x 200 ps, MD seed 0"},
        {"stage": "census_leads", "count": leads, "unit": "compositions",
         "detail": "census screen labels 'conditional' or 'review'"},
        {"stage": "replicates", "count": batches["replicate_seed12"], "unit": "trajectories",
         "detail": f"MD seeds 1 and 2 of the {leads} leads at 600/800/1000 K"},
        {"stage": "definitive", "count": batches["def_b1"] + batches["def_b2"] + batches["def_b3"],
         "unit": "trajectories",
         "detail": f"batches 1/2/3: {batches['def_b1']}/{batches['def_b2']}/{batches['def_b3']}; 400-1000 K, "
                   "200-800 ps, MD seeds 0-4"},
        {"stage": "classified", "count": sum(classes.values()), "unit": "compositions",
         "detail": "; ".join(f"{name} {classes[name]}" for name in
                             ("FAST_VALIDATED", "SLOW_MEASURED", "SLOW_ESTIMATED", "SLOW_BOUNDED",
                              "THERMALLY_UNSTABLE") if classes[name])},
        {"stage": "fast_conductors", "count": classes["FAST_VALIDATED"], "unit": "compositions",
         "detail": "FAST_VALIDATED: sigma(300 K) >= 0.1 mS/cm with replicate seeds"},
        {"stage": "unique_trajectories", "count": len(trajectories), "unit": "trajectories",
         "detail": "census + replicates + definitive campaigns (MACE-MP-0 small, NVT)"},
    ]
    return rows


def build() -> None:
    write_csv(paths.WORKFLOW, WORKFLOW_COLUMNS, workflow_rows())
    data.workflow.cache_clear()
