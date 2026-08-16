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

## Claim rule

A main rate claim needs the same ordered rate-quality trend on at least two task
families and both models, with all three seeds shown. A compression method must
beat the uniform code at a matched exact file rate, not only at a named width.
Cells stopped by either fixed gate cannot support the main claim. Report null,
harmful, screened, and failed cells with successful runs.
