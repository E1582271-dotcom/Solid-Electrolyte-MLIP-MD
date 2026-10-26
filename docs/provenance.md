# Provenance

## Where the numbers come from

This repository is a clean rebuild of a working repository in which the simulations were planned, run and analysed between September and October 2026. Results were not recomputed for the rebuild. [production/migrate_legacy.py](../production/migrate_legacy.py) read the committed analysis tables of that repository at commit `e2b7416c4af887fb9101aa444836164f689999a9` (clean working tree), rebuilt the classifier's run records with the original classifier code, and wrote the tables in [data/](../data/). [data/provenance/legacy_files.csv](../data/provenance/legacy_files.csv) maps every file here to the original files it came from, with their sha256.

The classifier of this repository ([src/ssemd/rules.py](../src/ssemd/rules.py)) contains the original rule functions unchanged; only the data loading around them was replaced. Run with the original file names as data sources, it reproduces the published classification table byte for byte, and the published summary except for the last bits of the fitted D0 values, which depend on the LAPACK build (relative difference below 1e-12; [data/frozen/](../data/frozen/), checked by `tests/test_classification.py`). The summary this repository writes rounds floats to 10 significant figures so that a rebuild gives the same bytes on any platform.

## Timeline of the classification protocol

| Date (UTC+8) | Event |
|---|---|
| 2026-09-08 to 10-15 | Seed-0 census (186 trajectories) and seed 1/2 replicates of the six census leads |
| 2026-10-24 00:56 | Protocol committed ([CLASSIFICATION_PROTOCOL.md](CLASSIFICATION_PROTOCOL.md)), before any new MD |
| 2026-10-24 01:26 | Amendment A1 (compute list and clarifications), before any new MD result |
| 2026-10-24 17:45 | Batch 1 analysed (29 runs); amendment A2 written before batch 2 ran |
| 2026-10-25 12:09 | Batch 2 analysed (45 runs); amendment A3 written after its results were seen and before batch 3 ran |
| 2026-10-26 | Batch 3 analysed (24 runs); every composition classified |

## Code and runtime identities

| Item | sha256 |
|---|---|
| MD code [production/md_runner/md.py](../production/md_runner/md.py), as run | `a773248ae302b8fec18c15cbf5caf78c3e7d21efc687c74ce94576eecc8aa0be` (the docstring was rewritten here; the code bytes are unchanged and tested) |
| MACE-MP-0 small model file | `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736` |
| Container image (PyTorch 2.5, CUDA 12.4) | `d7ec548f406de53d30078cb6caba2079afa1b2d02f597bf27b0edf6a45799dde` |
| MACE package tarball | `ebe6dde5a2b885ba8c1702adbd0d05cd1fedbd093766e4ba561e0cb3abd80ba7` |
| Initial-geometry guard [initialization_guard.py](../production/md_runner/initialization_guard.py) | `43ae4b13d1327421c69e882c83bb6aca3a4d7b22b3aaf32e2d404bfcf5de3f35` |
| kinisi settings of every definitive analysis | `3d5fa7290dbedae34f7322d2ea993d82efbcb30550938e80be526202d223ac60` ([analysis_settings.json](../data/provenance/analysis_settings.json)) |

Packages: mace-torch 0.3.16, torch 2.5.0, ase 3.29.0 for MD; Python 3.11 with numpy 2.4.6, scipy 1.17.1, kinisi 2.0.5, scipp 26.3.1, emcee 3.1.6 and pymatgen 2026.5.4 for analysis. GPUs: NVIDIA A100-SXM4-40GB of NSCC ASPIRE 2A.

## What was not carried over

- Raw trajectories and per-run posterior arrays (see [data.md](data.md)).
- Superseded and failed job-dispatch attempts, scheduler logs and job identifiers; their outcome is fully described by the task manifests in [production/manifests/](../production/manifests/).
- A supplementary second review of the 186 census trajectories (common-lag sampling, analysis-seed and covariance checks, framework review). It changed no census label and is superseded by the definitive classification.
- A July 2026 pilot that simulated four compositions picked by an earlier version of the Project 1 screen; it is superseded by the census of the full queue.

## Sanitisation

Site-specific paths, account names, host names and job identifiers were removed. The MD runner reads its site paths from `site.env` ([site.env.example](../production/md_runner/site.env.example)). `python -m ssemd check-repo` (run in CI) rejects CJK text, private identifiers and files larger than 5 MB.
