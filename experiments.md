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
controls. A main run writes ten codecs. A placement run writes three. The final
test sets never select a checkpoint, clipping threshold, or gate.

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

Encode every main adapter with uniform FP16, 8-, 4-, 3-, 2-, and 1-bit codes,
plus LoRAQuant `2@0.8`, `2@0.9`, `3@0.8`, and `3@0.9`. Uniform codes choose
among 99%, 99.9%, and 100% clipping on calibration NLL. LoRAQuant uses an SVD
split, group size 128, a one-bit low-energy part, and 100 update-error steps.

Measure exact file bits, effective bits per original adapter value, task score,
gain over the matched base model, and the fraction of the raw adapter gain that
the codec retains. Report all seeds, not only the Pareto points.

### Outputs

- `reports/natural_pareto.png`
- `reports/quantization_retention.png`
- `reports/summary.csv`

### Results

<!-- Fill after the campaign artifacts have been checked. -->

## Experiment 3: Exact rate accounting

### Question

How much does nominal value width understate the complete decoder-visible rate?

### Design and measures

For every `.fqcb` file, split the rate into header, packed values, FP16 scales,
byte padding, raw payload, compressed payload, and final file bits. Compare the
requested width with effective file bits per original LoRA value. This includes
all tensor names, shapes, split ranks, scales, and reconstruction metadata.

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

The nested mapping prefixes are 16, 32, 64, 128, 256, and 512. This produces
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

For every trained prefix, save the raw LoRA and run the adaptive 0/1/2/3/4/8-bit
MDL codec over target rates from 0.15 to 8 effective proxy bits/value. Decode
every `.fqmdl` file before evaluation. Define `R*(0.90)` as the smallest actual
serialized adapter file whose held-out accuracy retains at least 90% of the raw
adapter's accuracy gain over the base model.

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
# 2xA40 sanity check: random vs 64-bit structured task, seed 11.
bash scripts/run_information_scaling_a40.sh pilot

# Four information levels and three seeds.
bash scripts/run_information_scaling_a40.sh full
```

Configuration: `configs/information_scaling.yaml`.
Outputs: `runs_information_scaling/<mode>/prequential.csv`,
`adapter_information.csv`, per-prefix raw adapters and `.fqmdl` files, plus
`source_bits_vs_prequential_bits.png` and `source_bits_vs_adapter_bits.png`.

### Results

<!-- Fill after the pilot/full artifacts have been checked. -->

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
