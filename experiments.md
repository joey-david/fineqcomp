# Experiments

This protocol tests the project proposal's empirical bit-performance claim on
public tasks. It does not treat dataset byte size as task information and does
not use random labels. Dataset, model, and evaluator revisions are fixed in
`configs/campaign.yaml`.

The manifest has 48 candidate runs: two models, four tasks, three seeds, and two
adapter layouts. A run reaches the quantization sweep only after both fixed
calibration gates pass. Screened and failed-to-learn cells stay in the audit.

## Shared protocol

- Models: `mistralai/Mistral-7B-Instruct-v0.3` and `Qwen/Qwen3-8B`, with frozen
  NF4 backbones.
- Tasks: [GSM8K](https://huggingface.co/datasets/openai/gsm8k),
  [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa),
  [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc), and
  [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa).
- Seeds: 11, 22, and 33.
- Adapters: full rank-16 LoRA on all linear projections, and seeded-A rank-16
  LoRA on all Q/V projections.
- Stored precisions: 2, 3, 4, 8, and 16 bits. For each precision, clipping is
  selected from 99.0%, 99.9%, and 100.0% using calibration loss only.
- Rate: exact bits in the complete `.fqcb` file, including values, scales,
  shapes, names, reconstruction seed, and header. The frozen model is separate.
- General-skill check: official IFEval strict and loose prompt and instruction
  accuracy before and after adaptation.

GSM8K reserves a seeded 512-row slice of training for calibration and keeps its
test split final. CommonsenseQA has no labeled public test set, so a seeded
512-row training slice is calibration and the standard validation split is
final. ARC-Challenge and OpenBookQA use their standard validation and test
splits. No final test result affects task choice, early stopping, clipping, or
the learning gate.

Two gates run before the bit sweep:

1. Saturation gate: reject a model-task cell only when the lower end of its 95%
   Wilson interval exceeds 0.80 on calibration accuracy or exact match.
2. Learning gate: reject an adapter seed when the uncompressed adapter gains
   less than 0.05 on calibration data over its no-adapter baseline.

## Experiment 1: Real-task rate-performance curves

### Question

How many stored adapter bits are needed to obtain a useful held-out gain on
standard reasoning and knowledge tasks?

### Measures

- GSM8K final-answer exact match.
- CommonsenseQA, ARC-Challenge, and OpenBookQA constrained-choice accuracy.
- Test gain over the matched no-adapter model.
- Exact adapter file bits and raw packed payload bits.
- Paired bootstrap 95% interval and exact McNemar test against the matched base
  predictions.
- Training time and peak GPU memory.

### Outputs

- `reports/summary.csv`: every seed, adapter, task, and bit width.
- `reports/natural_pareto.png`: test gain against actual adapter MiB.
- `reports/baseline_screening.csv` and `reports/learning_gates.csv`: all fixed
  gate decisions, including rejected cells.

### Results

<!-- Fill after the new remote artifacts have been pulled and checked. -->

## Experiment 2: Quantization threshold

### Question

Where does reducing adapter precision cause a reliable loss relative to the
same trained 16-bit adapter?

### Measures

- Score at 2, 3, 4, and 8 bits divided by the matched 16-bit score.
- Paired score differences between each low-bit adapter and its matched 16-bit
  predictions.
- File-size reduction relative to 16 bits.
- Median and full seed range; no seed may be removed after the run.

### Outputs

- `reports/quantization_retention.png`: median retention and seed range by task,
  model, and adapter.
- `reports/summary.csv`: per-run paired statistics and storage fields.

### Results

<!-- Fill after the new remote artifacts have been pulled and checked. -->

## Experiment 3: Adapter allocation

### Question

At a matched stored size, does a full all-linear LoRA or a seeded all-layer Q/V
LoRA give the better task gain?

### Measures

- Pareto frontier of test gain versus exact file bits for each layout.
- Best task score under fixed 2, 5, 10, 20, and 50 MiB budgets where covered.
- Seed consistency across both model families and all tasks that pass both
  gates.

### Outputs

- `reports/natural_pareto.png` and the adapter fields in
  `reports/summary.csv`.

### Results

<!-- Fill after the new remote artifacts have been pulled and checked. -->

## Experiment 4: Task gain versus instruction retention

### Question

Does a smaller coded update retain more of the base model's instruction
following, and what task gain does that trade buy?

### Measures

- Test gain over the no-adapter task score.
- Drop in IFEval strict prompt accuracy from the matched no-adapter model.
- The same comparison for loose prompt and strict/loose instruction accuracy in
  `summary.csv`.

### Outputs

- `reports/ifeval_retention.png`: task gain against strict IFEval drop.

### Results

<!-- Fill after the new remote artifacts have been pulled and checked. -->

## Claim rule

The main claim needs the same bit-quality trend in at least two tasks and both
model families, with all three seeds shown. A single successful Mistral-GSM8K
cell remains a case study. Cells that fail either gate cannot support a
rate-performance claim. We will report null, harmful, and screened results next
to successful runs.
