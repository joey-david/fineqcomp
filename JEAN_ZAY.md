# Running fineQComp on Jean-Zay

Everything below runs from a login node after `ssh jean-zay`. The repo lives at
`$WORK/fineQComp`, so start with `cd $WORK/fineQComp`.

## The one thing that breaks jobs

`/etc/profile.d/z_modules.sh` exists on login nodes and **not** on `gpu_p6`
compute nodes. Sourcing it unconditionally killed fifteen jobs here in under
two seconds each, with an error no interactive test could reproduce.
`scripts/jean_zay_env.sh` now tries several module-init paths and fails loudly
if none works. If you write a new sbatch, source that script rather than
loading modules yourself.

Two more traps worth knowing:

- `--constraint=h100` is **mandatory**. Without it you get
  `Invalid job type for the account of the user` and no job.
- `/tmp` is node-local. Slurm `--output` paths must point at `$WORK`, or the
  log vanishes when the job ends.

## Choosing a queue

| QoS | Max walltime | Priority | Reality |
|---|---|---|---|
| `qos_gpu_h100-dev` | 2 h | **80** | Every job so far started in under 10 s |
| `qos_gpu_h100-t3` | 20 h | 50 | ~260 pending; waits of 10 min to 3.5 h |
| `qos_gpu_h100-t4` | 4 d | 45 | Longest jobs, lowest priority |

Dev also caps you at **10 submitted jobs** at once, counting pending ones.
Exceeding it gives `QOSMaxSubmitJobPerUserLimit`, and `sbatch --parsable` then
returns an empty job id, so a launch loop that chains dependencies silently
builds broken ones. Count the jobs before submitting a fan-out.

Dev is worth designing around. Three chained 2-hour dev stages usually finish
before a single long t3 job has started. Check your standing before assuming
priority:

```bash
idr_compuse            # want "under-consumption" for fas@h100
idr_quota_user         # disk
```

## Queue, watch, stop

```bash
# submit one stage of one seed
sbatch --export=ALL,SEED=11,STAGE=train scripts/jean_zay_pilot.sbatch

# chain a stage behind another job (afterany runs even if the first hit its wall)
sbatch --dependency=afterany:1234567 --export=ALL,SEED=11,STAGE=codecs \
  scripts/jean_zay_pilot.sbatch

# what is mine doing
squeue -u $USER -o "%.10i %.14j %.9T %.6M %.20R"

# why is it still pending, and when will it start
squeue -j 1234567 --start
scontrol show job 1234567 | grep -E "Reason|StartTime"

# stop things
scancel 1234567           # one job
scancel -u $USER          # everything of mine
scancel --name=fqpilot    # everything with that job name

# after it finishes: did it work, and how long did it take
sacct -j 1234567 -X -o JobID,State,Elapsed,ExitCode
sacct -u $USER -S now-7days -X -o JobID,JobName%16,QOS%20,Submit,Start,Elapsed,State
```

That last one is how to tell a queue wait from a crash: compare `Submit` with
`Start`. If they are seconds apart the queue was never the problem.

## Watching progress live

Progress bars go to stderr, ordinary prints to stdout:

```bash
JOB=1234567
tail -f slurm_logs/fqpilot-$JOB.err     # tqdm bars
tail -f slurm_logs/fqpilot-$JOB.out     # stage markers
```

tqdm redraws with carriage returns, which `tail` shows as one long line. This
turns it back into readable lines:

```bash
tail -f slurm_logs/fqpilot-$JOB.err | tr '\r' '\n'
```

Training writes a JSON line every 150 optimizer updates, which is the most
useful signal of all — watch `validation_nll` stop falling:

```bash
tail -f runs/pilot__*s11*/logs/training.jsonl
```

A compact live dashboard:

```bash
watch -n 30 '
  squeue -u '$USER' -o "%.10i %.14j %.9T %.6M"
  echo
  for d in runs/pilot__*/; do
    echo "$(basename $d): $(cat $d/status.json 2>/dev/null | tr -d "\n ")"
    echo "  codecs done: $(ls $d/codec_metrics 2>/dev/null | wc -l)"
  done'
```

## Getting results back to your laptop

From the laptop, not the cluster:

```bash
SSH_SERVER=jean-zay \
REMOTE_REPO_ROOT=/lustre/fswork/projects/rech/fas/uul94gf/fineQComp \
  ./scripts/remote.sh pull
```

`pull` skips checkpoints and codec files, so it stays small. Use `pull-stats`
for only the JSON and CSV.

## Staging data

Compute nodes have **no network**; login nodes reach the Hub through a proxy.
So datasets must be prepared on a login node before any job runs:

```bash
export repo_root=$PWD FINEQCOMP_ONLINE=1
source scripts/jean_zay_env.sh
$PYTHON -m fineqcomp prepare --config configs/pilot_metamath.yaml \
  --manifest prepared/pilot-metamath-manifest.jsonl
```

Models do not need downloading. Jean-Zay mirrors the Hub at
`$DSDIR/HuggingFace_Models`, and configs point straight at those directories.

## If a job dies

Work resumes at a useful granularity. Once `runs/<id>/raw_channel.pt` exists,
training is never repeated, and any codec whose `codec_metrics/<key>.json` is
already written is skipped. So resubmitting the same stage picks up where it
stopped. The exception is training itself: it is all-or-nothing, so it has to
fit inside one job's walltime.
