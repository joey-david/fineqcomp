# Experiments

This file fixes the campaign before remote execution. Every model and dataset
revision lives in `configs/campaign.yaml`. The manifest contains 96 training
runs; no-adapter evaluations are shared and do not add training jobs.

The adaptation rate always means the size in bits of the complete `.fqcb`
file. It includes packed values, FP16 row scales, tensor names and shapes, the
seeded-A reconstruction seed, and the file header. The frozen backbone is
reported separately and never counted as finetuning information.

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

## Experiment 2: Family, scale, and backbone checks

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

## Experiment 3: GSM8K and MBPP

### Question

Do actual coded adapter bits predict useful task gain on non-synthetic data,
and what instruction-following cost accompanies that gain?

### Setup

- Models: Qwen3-8B and Mistral-7B-Instruct-v0.3, both with NF4 backbones.
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

Natural dataset byte compression is descriptive only. It is not treated as
the task's true information content.

### Results

<!-- Leave empty until the remote artifacts have been pulled and checked. -->

## Fixed claim rule

The main claim requires a monotone rise in minimum coded bits with known task
information across the three mapping seeds. Unreached 70%, 90%, or 99%
accuracy targets remain censored. The analysis must not remove failed seeds,
adapter settings, or model checks after seeing their values.
