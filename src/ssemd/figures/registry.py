"""Every figure of the repository, in order. Each module exposes ``make() -> list[Path]``."""
from __future__ import annotations

import importlib
from pathlib import Path

# name -> (module, caption title)
FIGURES = {
    "fig1": ("fig1_workflow", "From 21,712 Materials Project entries to one class per composition"),
    "fig2": ("fig2_eligibility", "Per-temperature evidence behind every class"),
    "fig3": ("fig3_fast_arrhenius", "Arrhenius behaviour of the eleven fast conductors"),
    "fig4": ("fig4_sigma300", "Room-temperature conductivity estimates and bounds for all 62 compositions"),
    "fig5": ("fig5_robustness", "Robustness of the classification"),
    "ed01": ("ed01_msd", "Mean-squared displacement curves"),
    "ed02": ("ed02_atlas", "Atlas of the 62 compositions"),
    "ed03": ("ed03_census_flags", "Seed-0 census flags"),
    "ed04": ("ed04_windows", "Time-window and covariance sensitivity of the kinisi estimates"),
    "ed05": ("ed05_framework", "Framework integrity criteria"),
    "ed06": ("ed06_thermostat", "Temperature control and energy drift"),
    "ed07": ("ed07_six_leads", "Per-seed Arrhenius fits of the six census leads"),
    "ed08": ("ed08_screening", "Project 1 screening score against the final class"),
}


def build(only=None) -> list[Path]:
    written: list[Path] = []
    for name, (module, _) in FIGURES.items():
        if only and name not in only:
            continue
        written.extend(importlib.import_module(f"ssemd.figures.{module}").make())
    return written
