# Working with fineQComp

Run commands from the repository root. Python 3.11 or newer and `uv` are
required. Scoring saved results and rebuilding plots work on CPU; NF4 training
and model evaluation need a Linux NVIDIA GPU host. Model weights and prepared
corpora are not part of the repository.

## Reproduce the saved paper results

```bash
uv run scripts/reproduce.py
```

Outputs go to `.cache/reproduction/` (override with `--out DIR`): figure PDFs
in `figures/`, PNGs in `previews/`, the breadth table in `tables/`, and predictor
scores and predictions in `results/`. `summary.json` records the scope and
all scores. The command recomputes the original, three-seed replication,
cross-corpus audit, and disjoint-example validation reports with the existing
scorer. It fails if any report differs from its saved result beyond small
floating-point roundoff, or if a plot fails to build.

Other plots and the breadth table use the saved measured tables. The command
copies the stored layer-location PDF and intro diagram source, whose renders
it cannot rebuild. It does not rerun training or model evaluation, reconstruct
missing per-row predictions, or compile the final manuscript. The
[paper guide](../paper/README.md) lists each figure's inputs and source limits.

## Layout

| Path | Owns |
|---|---|
| `src/fineqcomp/` | Run contracts, data, training, codec, evaluation, and shared analysis |
| `src/fineqcomp/studies/` | Rate, recovery, spectral-filter, and payload studies |
| `configs/rate/` | Dataset/receiver panels and rate controls |
| `configs/recovery/` | Corruptions, matched controls, window searches, and transfer probes |
| `configs/payload/` | Synthetic source-information controls |
| `results/` | Recorded tables, locks, audits, and study notes |
| `scripts/analysis/` | CPU scoring and GPU profile helpers |
| `scripts/jean_zay/` | Explicit Slurm launchers |
| `paper/` | Submission PDF, local source snapshot, plots, and their measured inputs |
| `.cache/` | Local prepared data, run files, predictions, and scratch outputs; ignored |

`pyproject.toml` defines dependencies; `uv.lock` fixes their versions. There is
no second requirements list. The console command and `python -m fineqcomp`
call the same CLI. Use `uv run fineqcomp --help` for its commands.

## A campaign

Check a small manifest before downloading data or loading a model:

```bash
uv run fineqcomp prepare --config configs/rate/compressibility_smoke.yaml \
  --manifest .cache/prepared/smoke.jsonl --no-data
uv run fineqcomp run --config configs/rate/compressibility_smoke.yaml \
  --manifest .cache/prepared/smoke.jsonl --dry-run --limit 1
```

On the GPU host, prepare the pinned datasets by running the first command
without `--no-data`, then remove `--dry-run` from the second command to train
and score the selected run. `--pilot-rows 8` tests execution on a small sample;
it does not produce a paper result.

```bash
uv run fineqcomp analyze --root .cache/runs --out .cache/reports/smoke
```

A campaign expands into immutable `RunSpec` records. The manifest must match
the config before a run starts. Each run writes its config, metrics, predictions,
codec files, and status under its run ID. Rates include decoding metadata and
come from files that the evaluator reloads. No rate is inferred from a nominal
bit width. Raw gain, best-decoded gain, interpolated crossings, and first-feasible
file rates remain distinct fields in the recorded results.

## Study entry points

Study modules reuse the shared runner and codec. Their `--help` lists phases:

```bash
uv run python -m fineqcomp.studies.spectral_transfer --help
uv run python -m fineqcomp.studies.generalisation --help
uv run python -m fineqcomp.studies.matched_budget --help
uv run python -m fineqcomp.studies.information_scaling --help
```

Use `spectral_transfer` for search, independent finalist selection, and frozen
transfer tests; `generalisation` for fixed compression controls on existing
adapters; `matched_budget` for rank/precision and shrinkage controls; and
`information_scaling` for the known-payload control. Phase locks cover prepared
inputs and all package code, including study modules. Changed code or inputs
require a fresh study output directory. Historical locks are evidence, not
configs to edit until an old run resumes.

## Jean-Zay

The launchers take their working root from `SLURM_SUBMIT_DIR`. Preload
`scripts/jean_zay/env.sh` on the login node when compute nodes lack modules.
Set `HF_HOME` to the shared model cache and stage data before using offline mode.
Override the account with `sbatch --account=<project>@h100`. For H100 jobs under
two hours use partition `gpu_p6`, constraint `h100`, and QoS
`qos_gpu_h100-dev`; use `qos_gpu_h100-t3` for longer jobs. Use the matching A100
account, partition, constraint, and QoS when running on A100s.

`run.sbatch` accepts `CONFIG`, `MANIFEST`, `SHARDS`, `PREPARED_ROOT`, and
`RUNS_ROOT`; `spectral.sbatch` takes a phase followed by the study's CLI options.
Choose the time and array width from a completed small run. Launchers do not
automatically start dependent experiments.

`scripts/remote.sh` requires both `SSH_SERVER` and `REMOTE_REPO_ROOT`. Its
`--help` is local; every other action contacts the chosen host.

## Extending a study

Add a YAML campaign using the nearest existing config. Keep shared changes in
the module that already owns them: `data.py` for prepared rows and corruptions,
`codec.py` for serialized updates, `evaluation.py` for scores, and `runner.py`
for execution. A new study belongs in `studies/` only when existing phases
cannot express it. Search and selection must not read test outcomes. Report
missing, failed, and no-learning cells along with successes.

```bash
uv run pytest
uv run python paper/plots/make_figures.py --preview .cache/paper-preview
```

The [paper guide](../paper/README.md) maps figures to their inputs and explains
which manuscript source is available.
