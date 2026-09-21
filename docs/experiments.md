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

## Experiment 7: Known task information against adapter description length

### Question

At identical prompts, rows, and optimizer updates, does the smallest adapter
file that still reproduces a learned map grow with the information that map
contains?

Experiment 2 answered a weaker question. It showed R*(0.90) rising with the
number of distinct training rows at fixed compute, but never measured what
those rows contained, so "unique information" stayed a proxy for "unique rows".
Here the information is known by construction and a second, measured quantity
is checked against it.

### Controlled task

Every prompt names a family and an item and asks for one of sixteen
single-token labels, under one canonical template. There is one example per
mapping, so distinct rows and source symbols are the same thing. Rows, prompts,
model, adapter, optimizer, and optimizer updates are identical across
conditions. Only the labels change.

| condition | rule | source bits at 512 mappings |
|---|---|---:|
| `constant` | one label everywhere | 4 |
| `p1` … `p16` | K hidden 16-item prototype tables, reused across families | 64·K |
| `random` | an independent label per mapping | 2,048 |

`constant` is the zero-information anchor. Whatever R* costs there is the price
of addressing a behaviour at all, and every other condition is read against it.
Without that intercept a positive slope through points that all sit far above
zero is not evidence of anything.

The prefixes are 64, 128, 256 and 512 mappings, out of 768. Each prefix spends
the same 2,048 optimizer updates, with epochs derived, so a longer prefix buys
more information and not more gradient steps. A prefix that cannot hit the
budget exactly is an error rather than a rounding.

### What one cell measures

- **Source bits**, known by construction: four times the number of independent
  symbol draws the labels required.
- **A conditional prequential code**. The frozen base is shared side
  information and codes the first block. Every later block is coded by an
  adapter trained only on the blocks before it. No future block trains or
  selects any encoder.
- **R\*(0.90)**, the smallest adapter file that still reproduces the taught map,
  on a dense uniform ladder from an empty file through 2 bits per value.

The label code is a mixture, `(1-w)·p_model + w/16` at a pre-registered
`w = 1/16`, not the model's own renormalized distribution. A model that is
confidently wrong on an unseen mapping costs an unbounded number of bits under
its own probabilities, so the reported code length ends up set by the numerical
floor in the scorer rather than by the data. The mixture caps the cost per
symbol at 8 bits and adds at most 0.09 bits when the model is right.

The ladder carries an explicit empty-adapter point at zero file bits and zero
gain, so the crossing is always bracketed from below. R* is undefined, and
reported as undefined, unless the raw adapter clears an absolute gate of one
bit saved per mapping out of a maximum of four: below that every rung retains
ninety percent of nothing.

Every per-example probability is written to `predictions/*.jsonl`. A finished
run can then be recalibrated, re-coded, or broken down by family without
training anything again.

### Pre-registered gates

These are recorded before the run. They are pass/fail, and a failure is a
result rather than a reason to retune.

- **G1 — did the learner reach the map?** Raw recall on the taught mappings is
  at least 0.95 in every main cell. Below that R* is a ratio of noise.
- **G2 — is the code a code?** The cumulative prequential code never exceeds
  the cumulative base-model code. The base code is free, so a valid code cannot
  be worse; a violation is a bug in the measurement.
- **G3 — does the code see the source?** On unseen mappings the `random`
  condition costs near four bits each while `constant` and `p1` cost far less.
  This is what makes the code a measure of source information rather than of
  the base model's letter prior.
- **G4 — what is the noise floor?** At 64 mappings only four families exist, so
  `p4`, `p8`, `p16` and `random` are the same task: 64 arbitrary labels, 256
  source bits. Their R* must agree. Any slope claimed in G5 has to clear this
  spread.
- **G5 — the claim.** At 512 mappings R* is monotone in source bits and the
  extreme conditions do not overlap across seeds.

A failure of G5 with G1–G4 passing is a real negative, and an informative one:
it would say the adapter file is dominated by the cost of naming a point in
weight space rather than by task content, and it would reframe Experiment 2's
slope as something other than information.

### Studies

| study | question | cells |
|---|---|---:|
| `main` | R* against known source bits, four prefixes, rank 16 | 21 |
| `rank` | is R* task content or addressing cost? ranks 4 and 64 at 512 mappings | 24 |
| `cue` | does naming the shared structure in the prompt change what must be stored? | 6 |

The rank study is the sharpest test in the design. A GSM8K adapter needed about
one bit per value over 42M values — 42 megabits — for a task whose genuine
information content is plausibly kilobits. Almost none of that file is task
content. If R* in total file bits is flat in rank, the adapter stores task
information and the container adapts to it; if it scales with parameter count,
R* is measuring the cost of addressing a point in weight space, and the natural
campaign cannot mean what it appears to mean.

The cue study pairs `p4` and `p16` with versions whose prompt names the
prototype. Source bits are unchanged; what changes is whether the learner has
to discover the sharing rule itself. The gap is how much reusable structure the
optimizer fails to exploit when it is not told.

### Run

```bash
# Two cells: the zero-information anchor and the hardest condition, seed 11,
# largest prefix. Gate: raw recall >= 0.95 on both, and R* defined on both.
sbatch --array=0-1%2 --export=ALL,MODE=smoke \
  scripts/jean_zay_information.sbatch

# All 51 cells across twelve shards, about 45 minutes each.
sbatch --array=0-11%12 --export=ALL,MODE=full \
  scripts/jean_zay_information.sbatch

python -m fineqcomp.information_scaling --out runs_information_scaling/full --aggregate
```

Configuration: `configs/information_scaling.yaml`. Outputs:
`runs_information_scaling/<mode>/cells.csv`, `prequential.csv`, `gates.json`,
`information_scaling.png`, and per-cell `result.json`, raw checkpoints and
predictions.

### Results

<!-- Fill after the artifacts have been checked. -->

### Staged natural-task replication

Advancement gates are cheap filters, not claim tests. A cell advances when its
held-out response code improves by at least 0.02 bits/token and its task-score
point estimate improves by at least 10% of the available headroom. Its interval
need not exclude zero at the screen. A positive endpoint slope is enough to buy
the three-arm repeat. Final claims still need all seeds and uncertainty.

| stage | unit of work | expected time per H100 | parallel wall time |
|---|---|---:|---:|
| synthetic smoke | one condition, seed 11, largest prefix | 15 min | 15 min for two cells |
| synthetic full | one shard of four or five cells | 45 min | 45 min for twelve shards |
| NLL screen, XSum | one model, 125 updates | 12 min | 12 min for both models |
| NLL screen, Magicoder | one model, 125 updates | 35 min | 35 min for both models |
| 500-update extension, XSum | one model | 30 min | 30 min for two models |
| 500-update extension, Magicoder | one model | 1 h 45 min | 1 h 45 min for two models |
| natural endpoint, math | one arm, 2,000 updates and dense curve | 1 h 30 min | 1 h 30 min for selected arms |
| natural endpoint, XSum | one arm, 2,000 updates and dense curve | 1 h 45 min | 1 h 45 min for selected arms |
| natural endpoint, Magicoder | one arm, 2,000 updates and dense curve | 4 h | 4 h for selected arms |

These are launch budgets, not measured outcomes. Jean-Zay runs one shard on
each H100. The synthetic estimate scales the saved 53-minute, 2,000-update
Mistral training time to 2,048 short-sequence updates per prefix, allows five
minutes for the one model load a shard needs, and ten seconds per coded file:
the whole thirteen-rung ladder encodes and decodes in under ten seconds at rank
16, measured, so the budget is training. Long-context code gets the larger
bound. Each stage records its actual load, train, codec,
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
