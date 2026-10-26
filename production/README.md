# Production: MD runner and batch manifests

Code that produced the raw data. None of it is needed to rebuild the tables or figures of this repository, and none of it runs in CI.

| Path | Content |
|---|---|
| `md_runner/md.py` | NVT Langevin MD with MACE-MP-0 (the code that ran; only its docstring differs) |
| `md_runner/run_one.py` | runs one task: checks model, code, input structure and protocol hashes, runs MD, validates the raw output (frame count, step clock, finite values, temperature and energy agreement) and writes its sha256 |
| `md_runner/task.sh` | per-task environment (container, packages, model) and an atomic claim so that any number of pools can share one task list |
| `md_runner/pool.pbs`, `pool_multinode.pbs`, `node_pool.sh`, `run.pbs` | PBS GPU pools: one worker per GPU, longest task first, per-task walltime fit; the multi-node pool starts one node pool per node |
| `md_runner/initialization_guard.py` | verifies that continued replicates start from the canonical geometry of earlier seeds |
| `md_runner/site.env.example` | site paths (copy to `site.env`) |
| `manifests/<batch>.json` | every task of a batch (composition, temperature, seed, steps, protocol hash, trajectory sha256) |
| `initializations/` | canonical initial geometries of the two leads continued with seeds 3 and 4 |
| `make_campaign.py` | assembles a runnable campaign folder from a manifest |
| `migrate_legacy.py` | one-off import of the analysed results from the original working repository |

## Running a batch

```bash
python production/make_campaign.py production/manifests/def_b3.json /path/to/campaign
cd /path/to/campaign && cp site.env.example site.env   # edit the paths
qsub -P $PROJECT -l select=1:ncpus=64:mem=440gb:ngpus=4 -l walltime=24:00:00 pool.pbs
python3 run_one.py --summary                           # status of every task
```

The census and replicate manifests (`census_12`, `census_50`, `replicate_seed12`) were reconstructed from the analysed results; they ran with an earlier runner that used the same MD code and protocol. For the definitive batches, `make_campaign.py` rebuilds exactly the protocol hashes recorded when they ran.
