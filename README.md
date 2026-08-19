# fineQComp

### From reasoning trajectories to information budgets

How many transmitted bits of a learned update are actually needed to preserve
new model behavior? fineQComp measures task gain against the exact serialized
size of LoRA updates.

## Fast Pareto sweep

Reuse an already-trained run instead of training again:

```bash
.venv/bin/python -m fineqcomp.pareto runs_literature/<run-id>
```

The default sweep evaluates a single zero-free quantizer family at 1, 2, 3, and
4 nominal bits. Its reconstruction levels are symmetric odd levels
(`±1, ±3, ...`) with an MSE-refit scale per row, so small values are never
rounded to a dedicated zero code. The 1-bit member is exactly the existing
binary sign × mean-absolute-value quantizer and is reused when its result is
already present; only 2/3/4-bit need new task evaluations.

Results go to `<run>/pareto/`:

- `pareto.png` — effective bits/value vs capped retained task gain, with the
  Pareto frontier highlighted.
- `pareto.csv` / `pareto.json` — exact rates, task scores, retained gain, and
  quantization error.
- `midrise*.json` / predictions — restart-safe per-point artifacts.

Use `--bits 1 2 3 4 8` for the longer sweep or `--force` to recompute points.
The x-axis is **effective transmitted bits/value**, including scales, headers,
padding, and zlib compression—not nominal quantizer width.

## Full campaign

```bash
./scripts/bootstrap.sh
.venv/bin/python -m fineqcomp prepare
.venv/bin/python -m fineqcomp run --shard 0 --shards 2
.venv/bin/python -m fineqcomp analyze
```

The fixed campaign trains rank-16 adapters on MetaMathQA, Magicoder, and XSum
with Mistral-7B and Qwen2.5-7B, then evaluates GSM8K, MATH, HumanEval, and XSum.
Runs are restart-safe and preserve the raw adapter so codec experiments can be
repeated without retraining.
