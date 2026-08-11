# Experiments

This file fixes the campaign before remote execution. Every model and dataset
revision lives in `configs/campaign.yaml`. The manifest contains 105 candidate
runs; no-adapter evaluations are shared and do not add training jobs.

The adaptation rate always means the size in bits of the complete `.fqcb`
file. It includes packed values, FP16 row scales, tensor names and shapes, the
seeded-A reconstruction seed, and the file header. The frozen backbone is
reported separately and never counted as finetuning information.

## Eight-worker mixed-GPU execution

Run preparation once on a host with access to the pinned Hugging Face snapshots:

```bash
cd ~/fineQComp
set -a; source .env; set +a
"$PYTHON" -m fineqcomp prepare --config configs/campaign.yaml
```

Make the same checkout, `prepared/`, model cache, and `runs/` directory visible
on all four hosts. Stop older campaign workers first: every live worker must use
the shared per-run lock in this revision. From one node with SSH aliases for the
four hosts, start the full grid with:

```bash
./scripts/run_distributed_campaign.sh
```

Use `./scripts/run_distributed_campaign.sh --prepare` when preparation has not
been run on the launch node. The script makes one SSH connection per host and
creates one two-window tmux session per host. It leaves an existing
`fineqcomp8` session untouched, so rerunning it cannot start duplicate workers.

All workers read the same manifest and take an exclusive lock before touching a
run. Completed runs remain skipped, so this layout keeps valid artifacts from
earlier launches. `upnquick` and `ourasi` run the full grid. The 11 GB cards on
`boldeagle` and `readycash` run only 7B or 8B NF4 jobs of at most 192 tokens at
micro-batch 1. Gradient accumulation keeps the effective batch size fixed.
Those older cards use FP16 compute because they lack native BF16.

After all eight workers finish, run `"$PYTHON" -m fineqcomp analyze` once from
the shared checkout.

## Jean-Zay H100 execution

Jean-Zay uses three restart-safe stages. `jean_zay_stage.sbatch` runs on a
networked pre/post node, creates a system-site virtual environment, installs the
pinned IFEval evaluator, prepares every dataset, and downloads every pinned
model snapshot. `jean_zay_smoke.sbatch` then checks one H100 and completes one
small campaign run. Only after that succeeds should `jean_zay_array.sbatch` be
submitted: its 16 one-GPU tasks map array indices 0 through 15 to the same 16
global manifest shards. `jean_zay_analyze.sbatch` runs after the complete array.

The H100 jobs load the managed PyTorch 2.8/CUDA 12.8 module and work offline.
All jobs share `$WORK/fineQComp/{prepared,.hf_cache,runs,reports}`. Failed or
preempted array elements may be resubmitted without repeating completed runs.

## Experiment 1: Known-information rate-distortion law

### Question

Does the smallest quantized update needed for a fixed error grow with the
number of independent bits in the finetuning task?

### Setup

- Model: `Qwen/Qwen3-8B-Base`, NF4 backbone, BF16 computation.
- Data: 16,384 rows at each `K` in `{4, 32, 256, 1024}` and mapping seeds
  `{11, 22, 33}`.
- Each family contains 16 independent labels from a 16-token alphabet. The
  source has `16 K` four-bit symbols and exact entropy `64 K` bits.
- Three fixed random codebook assets supply the labels. Each asset stores the
  full 65,536-bit source rather than a short PRNG seed; smaller `K` settings
  use a hashed prefix of the same source.
- Training and test use different wording and nuisance IDs. Both query the same
  family-item mapping.
- Seeded-B settings: last-layer Q/V rank 4; last-four Q/V rank 4; last-four Q/V
  rank 16; all-layer Q/V rank 16.
- Full-LoRA control: all linear projections, rank 16, at `K=32` and `K=1024`.
- Adapter precisions: 2, 3, 4, 8, and 16 bits. Select clipping from
  `{99.0, 99.9, 100.0}` using calibration NLL only.

### Metrics and outputs

- Primary: unique family-item error and the 16-way Hamming lower bound
  `16 K [4 - h2(D) - D log2(15)]`.
- Accuracy-target plots show each seed plus the median and seed range. A failed
  target stays marked as right-censored at the largest tested channel.
- Secondary: accuracy, label NLL, calibration error, actual file bits, raw
  payload bits, BF16 nominal bits, training time, and peak memory.
- Plots: `rate_distortion.png`, `bits_vs_information.png`, and
  `rate_allocation.png`.

### Results

<!-- Leave empty until the remote artifacts have been pulled and checked. -->

## Experiment 2: Controlled transfer through real paraphrases

### Question

Does the rate-distortion relation survive when the identifiers are natural
sentences and test queries are unseen paraphrases?

### Setup

- Source: positive paraphrase pairs from the pinned PAWS training split.
- Deterministic filtering removes empty, identical, or reused sentences.
- The first sentence supplies training and clipping-calibration queries under
  distinct wrappers. Only its paired sentence supplies the transfer test.
- Each pair receives one independent label from the same fixed 16-token random
  codebooks as Experiment 1.
- Binding counts are `{256, 2048, 8192}`, with seeds `{11,22,33}` and 8,192
  total training rows per cell.
- Model: Qwen3-8B-Base/NF4 with seeded-B last-four Q/V rank 16.

### Metrics and outputs

- Unseen-paraphrase accuracy, distortion, label NLL, coded bits, and the 16-way
  Hamming lower bound.
- Plot: `controlled_transfer.png`.

### Results

<!-- Leave empty until the remote artifacts have been pulled and checked. -->

## Experiment 3: Family, scale, and backbone checks

### Question

Does the measured relation depend on Qwen3-8B, NF4, or one model size?

### Setup

- Mistral-7B-v0.3/NF4: `K={32,1024}`, three seeds, seeded-B last-four rank 16
  and full all-linear rank 16.
- Qwen3-14B-Base/NF4: the same two `K` values and adapter settings, seed 11.
- Qwen3-8B-Base/BF16: the same two `K` values, seeded-B last-four rank 16,
  seed 11.
- All data, prompts, codec settings, and metrics match Experiment 1.

### Metrics and outputs

- Compare rate at matched distortion, rate divided by the lower bound, and
  whether family or scale changes the slope.
- Plot: `model_checks.png`; table: `efficiency.csv`.

### Results

<!-- Leave empty until the remote artifacts have been pulled and checked. -->

## Experiment 4: Screened GSM8K and MBPP

### Question

Do actual coded adapter bits predict useful task gain on non-synthetic data,
and what instruction-following cost accompanies that gain?

### Setup

- Models: Qwen3-8B and Mistral-7B-Instruct-v0.3, both with NF4 backbones.
- Before training, both models run the complete held-out task evaluations.
  GSM8K uses a 0.80 exact-match ceiling and MBPP a 0.75 pass@1 ceiling. A cell
  is marked `too_easy` and receives no finetuning only when the 95% Wilson
  interval's lower bound exceeds its ceiling. These rules are fixed before
  results are available.
- Qwen uses seeds `{11,22,33}`; Mistral uses seed 11 as a family check.
- Adapters: seeded-B last-four Q/V rank 16, seeded-B all-layer Q/V rank 16,
  and standard full all-linear rank 16.
- GSM8K trains for three epochs and uses final-answer exact match.
- MBPP trains for ten epochs and uses pass@1 under a bounded Python evaluator.
- IFEval measures strict and loose prompt- and instruction-level retention.
- All adapters are evaluated at 2, 3, 4, 8, and 16 coded bits.

### Metrics and outputs

- GSM8K: exact match, held-out NLL, coded bits, and bootstrap interval.
- MBPP: pass@1, compile/rejection/timeout/test-failure counts, coded bits, and
  bootstrap interval.
- IFEval: strict and loose prompt and instruction accuracy before and after
  adaptation.
- Plots: `natural_pareto.png` and `ifeval_retention.png`.
- Table: `baseline_screening.csv`, including baseline score, headroom, threshold,
  and the pre-registered screening decision.

Natural dataset byte compression is descriptive only. It is not treated as
the task's true information content.

### Results

<!-- Leave empty until the remote artifacts have been pulled and checked. -->

## Fixed claim rule

The main claim requires a monotone rise in minimum coded bits with known task
information across the three mapping seeds. Unreached 70%, 90%, or 99%
accuracy targets remain censored. The analysis must not remove failed seeds,
adapter settings, or model checks after seeing their values.
Natural-task claims use only model-dataset cells marked `usable` by the fixed
base-score screen. Screened-out cells remain in the baseline audit and cannot
support either a gain or no-gain claim.
