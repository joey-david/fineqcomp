# Experiments

The campaign tests how many serialized adapter bits are needed to retain useful
fine-tuning gains. It follows the model, data, optimizer, adapter, and codec
settings in LoRAQuant, then adds the exact rate and behavioral information
measures called for by *How Many Bits Can an Adapter Write?* All Hub revisions
and fixed gates live in `configs/campaign.yaml`.

## Fixed protocol

- Frozen bases: `mistralai/Mistral-7B-v0.1` and `Qwen/Qwen2.5-7B`, loaded in
  double-quantized NF4.
- Main adapter: rank-16 full LoRA on every attention and MLP projection.
- Seeds: 11, 22, and 33.
- Training: two epochs, AdamW betas `(0.9, 0.95)`, learning rate `2e-4`, cosine
  decay, 30% warmup, global batch 16, no weight decay, and gradient norm 1.
- Data: MetaMathQA to GSM8K and MATH; Magicoder to HumanEval; XSum to XSum.
- Length: 1,024 tokens for math and XSum; 4,096 tokens for code.
- Standard metrics: exact answer with `math-verify`, HumanEval pass@1 with a
  bounded Python worker, and ROUGE-L with `rouge-score`.
- Raw checkpoint rule: train once, save one raw adapter, and derive every coded
  point from that same checkpoint.
- Gate rule: stop a cell if the base task score exceeds its fixed ceiling or if
  the raw adapter reduces held-out NLL by less than 0.02 bits per token.

The active manifest has 24 raw training runs: 18 main cells and six placement
controls. A main run writes twelve codecs. A placement run writes three. The
final test sets never select a checkpoint, quantizer, or gate.

## Experiment 1: Real-task learning and headroom

### Question

Do the two base models have enough headroom, and does each raw adapter learn its
training task before compression?

### Design and measures

Run all three task families on both models and all three seeds. Record the base
GSM8K exact match, HumanEval pass@1, or XSum ROUGE-L. Record the raw adapter's
held-out NLL gain in bits per token and its final task gain. For MetaMath, also
report MATH exact match as a second held-out measure.

### Outputs

- `reports/baseline_screening.csv`
- `reports/learning_gates.csv`
- raw task predictions in each run's `predictions/raw_task.jsonl`

### Results

<!-- Fill after the campaign artifacts have been checked. -->

## Experiment 2: Adapter rate-distortion curve

### Question

How much of the raw task gain survives at each exact serialized adapter rate?

### Design and measures

Encode every main adapter at FP16, 8, 4, 3, 2, and 1 bits with the zero-free
mid-rise quantizer, plus LoRAQuant `2@0.8`, `2@0.9`, `3@0.8`, and `3@0.9`.
LoRAQuant uses an SVD split, group size 128, a one-bit low-energy part, and 100
update-error steps.

Also encode `midtread2` and `midtread3`: the same widths under an absmax
quantizer with an exact zero level. These are matched-payload controls for
Experiment 2b, not baselines for the main frontier.

The mid-rise scale is fit by least squares per row, so there is no clipping
percentile to select and no calibration pass is spent choosing one.

Measure exact file bits, effective bits per original adapter value, relative
weight RMSE, task score, gain over the matched base model, and the fraction of
the raw adapter gain that the codec retains. Report all seeds, not only the
Pareto points.

### Outputs

- `reports/natural_pareto.png`
- `reports/quantization_retention.png`
- `reports/summary.csv`

### Results

<!-- Fill after the campaign artifacts have been checked. -->

## Experiment 2b: Code geometry against bit width

### Question

At a matched rate, does the shape of the codebook change retained gain as much
as the number of bits does?

### Design and measures

Compare `uniform2`/`uniform3` (zero-free mid-rise) against `midtread2`/
`midtread3` (exact zero level, absmax scale) on the same raw checkpoints. Both
pack the same payload bits per value. Report exact file bits for each, because
a code that emits few distinct symbols leaves more for zlib to remove, so the
two do not land at identical file rates.

The pre-registered expectation, from the reconstruction error alone: mid-tread
at two bits is worse than mid-rise at one bit, at a comparable file rate. If
retained task gain follows that ordering, then a bit width does not identify a
code, and rate-distortion claims stated in nominal bits are underspecified.

### Outputs

- `reports/quantization_retention.png`
- the `quantizer` and `relative_rmse` columns in `reports/summary.csv`

### Results

<!-- Fill after the campaign artifacts have been checked. -->

## Experiment 3: Exact rate accounting

### Question

How much does nominal value width understate the complete decoder-visible rate?

### Design and measures

For every `.fqcb` and `.fqmdl` file, split the rate into header, packed values,
FP16 scales, byte padding, raw payload, compressed payload, and final file bits.
Compare the requested width with effective file bits per original LoRA value.
This includes all tensor names, shapes, split ranks, row selectors, scales, and
reconstruction metadata.

Every evaluated point is decoded from its stored file before evaluation, and the
decoded reconstruction error is checked against the error the encoder reported.

### Outputs

- storage columns in `reports/summary.csv`
- the reloadable files in each run's `codecs/` directory

### Results

<!-- Fill after the campaign artifacts have been checked. -->

## Experiment 4: Behavioral information

### Question

Does artifact rate track how many prediction bits the adapter writes on train
and held-out examples?

### Design and measures

On fixed sets of 256 train and 256 held-out examples, compute token NLL under
the base, raw, and coded adapters. Report train and held-out code bits saved,
bits saved per token, and the train-minus-held-out excess. Compare held-out bits
saved with exact artifact bits for each task and codec.

### Outputs

- `reports/behavioral_write.png`
- behavioral information columns in `reports/summary.csv`

### Results

<!-- Fill after the campaign artifacts have been checked. -->

## Experiment 5: Matched-budget placement

### Question

At the same nominal LoRA parameter budget, does attention-only or MLP-only
placement write task information more efficiently than all-linear placement?

### Design and measures

On Mistral and MetaMath, compute separate attention-only and MLP-only ranks from
the model's actual projection shapes so each matches the all-linear rank-16
parameter count at the nearest integer rank. Set LoRA alpha to twice the computed rank. Run three seeds and
store FP16, LoRAQuant `2@0.8`, and `3@0.9`. Compare actual trainable parameters,
file bits, task gain, retained gain, and behavioral bits.

### Outputs

- `reports/placement_control.png`
- adapter rank and trainable parameter fields in each run's metrics

### Results

<!-- Fill after the campaign artifacts have been checked. -->

## Experiment 6: Cost and reproducibility

### Question

Can each result be reloaded and reproduced from its pinned inputs and coded
artifact, and what compute does it cost?

### Design and measures

Record model and data revisions, run seed, training updates, wall time, peak GPU
memory, and full codec settings. A run counts as complete only when every coded
file, metric record, and prediction file exists. Decode each adapter before its
evaluation. Keep screened, failed, and no-learning cells in the status report.

### Outputs

- `prepared/manifest.jsonl`
- per-run `config.json`, `training_metrics.json`, `codec_metrics/`, and
  `status.json`
- `reports/summary.json`

### Results

<!-- Fill after the campaign artifacts have been checked. -->

## Experiment 7: Dataset information vs learned adapter information

### Question

Holding dataset size and the training protocol fixed, does the amount of
independent information in the finetuning task predict (1) its conditional
prequential code length and (2) the minimum serialized adapter description
needed to retain the learned behavior?

### Controlled task

Every prompt identifies a family and one of 16 items and asks for one of 16
single-token labels. The visible prompts, number of rows, model, adapter, and
optimizer are matched between conditions. Only the source of the labels changes.

- `random`: every family-item mapping receives an independent random label,
  contributing exactly 4 new source bits per mapping.
- `structured_p1`: one random 16-item prototype table is reused by every family;
  task information saturates at 64 bits.
- `structured_p4`: four prototype tables repeat; task information saturates at
  256 bits.
- `structured_p16`: sixteen prototype tables repeat; task information saturates
  at 1,024 bits.

The nested mapping prefixes are 32, 128, and 512. This produces
conditions with identical example counts but very different known source
information. The prototype reuse rule is shared side information; only sampled
prototype labels count as task-specific source bits.

### Conditional prequential code

Use the frozen pretrained model as shared side information. Encode the first
block from its constrained 16-label probabilities. For every later block,
reset a rank-16 LoRA to the same deterministic initialization, train only on the
preceding prefix, and encode the unseen next block using the trained model's
16-label probabilities. Sum `-log2 p(label | prompt)` over blocks. No future
block is used for training or checkpoint selection.

This code intentionally scores one label per independent mapping rather than
all duplicated training rows, so repeated prompt scaffolding does not dominate
the dataset-compressibility measure.

### Adapter description length

For every trained prefix, save the raw LoRA and run one dense uniform-code
family from 0.0625 to 2 nominal bits per value. Decode every `.fqcb` file before
evaluation. Select the curve on a calibration split and report its chosen real
file on a separate test split. The utility ceiling is the best decoded
calibration point, including the raw adapter, so regularizing compression cannot
produce retention above 100%. Define `R*(0.90)` by interpolation on the upper
curve of response-code bits saved against the frozen base.

Primary plots:

1. known independent task bits vs conditional prequential code bits;
2. known independent task bits vs `R*(0.90)`;
3. conditional prequential code bits vs `R*(0.90)`.

The decisive pattern is not merely that larger datasets require larger
adapters: at the same mapping count, structured conditions should become cheap
once their prototype table has been learned, while the random condition should
continue to grow with its genuinely new source bits.

### Run

```bash
# Two-cell H100 check: random vs 64-bit structured task, seed 11.
sbatch --array=0-1%2 --export=ALL,MODE=pilot \
  scripts/jean_zay_information.sbatch

# Repeat only the two endpoint conditions on seeds 22 and 33.
sbatch --array=0-3%4 --export=ALL,MODE=repeat \
  scripts/jean_zay_information.sbatch

# Expand to four information levels only after checking the two short stages.
sbatch --array=0-11%12 --export=ALL,MODE=full \
  scripts/jean_zay_information.sbatch
```

Configuration: `configs/information_scaling.yaml`.
Outputs: `runs_information_scaling/<mode>/prequential.csv`,
`adapter_information.csv`, per-prefix raw adapters and `.fqcb` files, plus
`source_bits_vs_prequential_bits.png` and `source_bits_vs_adapter_bits.png`.

### Staged natural-task replication

Advancement gates are cheap filters, not claim tests. A cell advances when its
held-out response code improves by at least 0.02 bits/token and its task-score
point estimate improves by at least 10% of the available headroom. Its interval
need not exclude zero at the screen. A positive endpoint slope is enough to buy
the three-arm repeat. Final claims still need all seeds and uncertainty.

| stage | unit of work | expected time per H100 | parallel wall time |
|---|---|---:|---:|
| synthetic pilot | one condition, seed 11, three prefixes | 32 min | 32 min for two cells |
| synthetic repeat | one condition and seed | 32 min | 32 min for four cells |
| synthetic full | one condition and seed | 32 min | 32 min for twelve cells |
| NLL screen, XSum | one model, 125 updates | 12 min | 12 min for both models |
| NLL screen, Magicoder | one model, 125 updates | 35 min | 35 min for both models |
| 500-update extension, XSum | one model | 30 min | 30 min for two models |
| 500-update extension, Magicoder | one model | 1 h 45 min | 1 h 45 min for two models |
| natural endpoint, math | one arm, 2,000 updates and dense curve | 1 h 30 min | 1 h 30 min for selected arms |
| natural endpoint, XSum | one arm, 2,000 updates and dense curve | 1 h 45 min | 1 h 45 min for selected arms |
| natural endpoint, Magicoder | one arm, 2,000 updates and dense curve | 4 h | 4 h for selected arms |

These are launch budgets, not measured outcomes. Jean-Zay runs one cell on each
H100. The synthetic estimate scales the saved 53-minute, 2,000-update Mistral
training time to 672 short-sequence updates, then allows 14 minutes for model
load and the dense decoded-file sweep. Its batch limit is one hour. Long-context
code gets the larger bound. Each stage records its actual load, train, codec,
and task time so the next estimate can replace these budgets.

Natural arms use exact 32,000-example draw streams at 2k, 4k, 8k, 16k, and 32k
distinct rows. Within a data seed the sets are nested; the stop, rate-selection,
report, and task sets stay fixed. First run 2k/32k at seed 11, then 2k/8k/32k at
seeds 11 and 22, then all five arms at seeds 11/22/33. Only selected cells reach
the last stage. The main analysis fits a within-cell slope against conditional
prequential code length; adjacent arms may invert.

### Results

<!-- Fill after the pilot/full artifacts have been checked. -->

## Directional pilot and its gates

Before any multi-day campaign, one cheap experiment tests whether the
measurement chain works and whether the codec ordering is real. Config:
`configs/pilot_gsm8k.yaml`. Qwen2.5-1.5B on an NF4 base, GSM8K's own 7K train
split, three seeds, the full codec grid plus the adaptive frontier, scored on a
fixed 400-problem subsample of the GSM8K test set.

These three gates are recorded before the run. They are pass/fail, and a failure
is a result, not a reason to retune.

- **Gate 1 — is there anything to compress?** The raw adapter must clear the
  fixed learning gate of 0.02 bits per token of held-out NLL gain, on all three
  seeds. If it fails, no downstream number means anything.
- **Gate 2 — does weight error predict behavior?** `midtread2` must retain no
  more task gain than `binary` at a comparable exact file rate. This follows
  from the measured reconstruction error, so failing it means task score does
  not track weight error, and the rate-distortion framing needs rethinking
  before scale-up.
- **Gate 3 — does adaptive allocation earn its place?** At least one adaptive
  MDL point must dominate every fixed-rate point at matched or lower exact file
  bits, somewhere below 2 bits per learned scalar. If it fails, the honest
  reading is that per-row weight MSE is the wrong allocation signal, and the
  behavior-aware allocator should be built before the campaign, not after.

Gate 3 is the one that authorizes the full campaign. Gates 1 and 2 only
establish that the pilot measured anything at all.

A pilot result does not support any claim about the 7B NF4 regime. The pilot
model is small and its base is not in the bit-constrained regime the proposal
is about, so a passing pilot buys one 7B confirmation cell, not the campaign.

### Results

<!-- Fill after the pilot artifacts have been checked. -->

## Claim rule

A main rate claim needs the same ordered rate-quality trend on at least two task
families and both models, with all three seeds shown. A compression method must
beat the uniform code at a matched exact file rate, not only at a named width.
Cells stopped by either fixed gate cannot support the main claim. Report null,
harmful, screened, and failed cells with successful runs.

For Experiment 7, the synthetic source-bit claim additionally requires the
expected ordering to replicate across all three seeds and requires raw adapters
to learn each measured prefix well enough that `R*(0.90)` is defined. The
synthetic calibration can establish that the measurement tracks known task
information; claims about natural datasets still require a separate
prequential-code analysis on the real task families.
