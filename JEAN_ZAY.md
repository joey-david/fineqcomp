# Jean-Zay

Repo lives at `$WORK/fineQComp`. Everything below runs from a login node.

## Four things that break jobs

- **`/etc/profile.d/z_modules.sh` does not exist on compute nodes.** Sourcing it
  killed fifteen jobs in under two seconds each, with an error no interactive
  test reproduces. Source `scripts/jean_zay_env.sh` instead of loading modules
  yourself; it tries several init paths and fails loudly.
- **`--constraint=h100` is mandatory.** Without it: `Invalid job type for the
  account of the user`.
- **`/tmp` is node-local.** Slurm `--output` paths must point at `$WORK` or the
  log vanishes with the node.
- **dev caps you at 10 *submitted* jobs.** Past that, `sbatch --parsable`
  returns an empty id and a dependency chain silently builds broken links.

## Queues

| QoS | Walltime | GPUs | Submitted | Priority | Reality |
|---|---|---|---|---|---|
| `qos_gpu_h100-dev` | 2 h | 32 | 10 | 80 | starts in seconds |
| `qos_gpu_h100-t3` | 20 h | 512 | 10000 | 50 | waits 10 min to 3.5 h |
| `qos_gpu_h100-t4` | 4 d | 64 | 1000 | 45 | longest, lowest priority |

Check standing with `idr_compuse` (want "under-consumption") and `idr_quota_user`.

## Watching

```bash
scripts/monitor.sh                 # queue, per-cell progress, crashes
scripts/monitor.sh fqcomp          # one job name
watch -n 60 scripts/monitor.sh     # refresh in place
```

Raw alternatives:

```bash
squeue -u $USER -o "%.10i %.8j %.9T %.7M %.7L %.14R"
squeue -j JOBID --start                       # why pending, when starting
sacct -j JOBID -X -o JobID,State,Elapsed,ExitCode
tail -f slurm_logs/fqcomp-JOBID.err | tr '\r' '\n'   # tqdm, made readable
tail -f runs/<cell>/logs/training.jsonl              # validation_nll per checkpoint
```

`sacct` comparing `Submit` with `Start` is how you tell a queue wait from a
crash. If they are seconds apart, the queue was never the problem.

## Submitting and stopping

```bash
sbatch --export=ALL,ARM=arm_a,SEED=11 scripts/jean_zay_compress.sbatch
sbatch --dependency=afterok:JOBID --export=ALL,... scripts/...   # chain
scancel JOBID | scancel -u $USER | scancel --name=fqcomp
```

`afterany` runs the next stage even if the previous one hit its wall; `afterok`
stops the chain on failure. Use `afterany` for a report, `afterok` for work that
depends on the previous stage having succeeded.

## Data

Compute nodes have **no network**; login nodes reach the Hub through a proxy, so
datasets must be staged before any job runs:

```bash
export repo_root=$PWD FINEQCOMP_ONLINE=1
source scripts/jean_zay_env.sh
$PYTHON -m fineqcomp prepare --config configs/<name>.yaml --manifest prepared/<name>.jsonl
```

Models need no download: configs point straight at `$DSDIR/HuggingFace_Models`.

## Results back to your laptop

```bash
SSH_SERVER=jean-zay \
REMOTE_REPO_ROOT=/lustre/fswork/projects/rech/fas/uul94gf/fineQComp \
  ./scripts/remote.sh pull-stats     # json/csv only;  pull-logs for slurm logs
```

## When a job dies

Resubmitting resumes: once `runs/<id>/raw_channel.pt` exists training is never
repeated, and any codec with a metric file on disk is skipped. Training itself
is all-or-nothing, so it must fit one job's walltime.
