# fineQComp

### From reasoning trajectories to information budgets

How many transmitted bits of a learned update are actually needed to preserve
new model behavior? fineQComp measures task gain against the exact serialized
size of LoRA updates.

## Fast Pareto sweep

Reuse an already-trained run instead of training again. For the LAMSADE setup,
the distributed launcher evaluates one point on each GPU: 2/3-bit on the two
`upnquick` A100s and 4/8-bit on the two `ourasi` A6000s. The existing binary
result supplies the mathematically identical 1-bit point.

```bash
RUN=runs_literature/<run-id>
OLD_SESSION=fineqcomp-lit-0817-1137 bash scripts/run_pareto_distributed.sh "$RUN"
```

`OLD_SESSION` is optional; when set, that tmux campaign is stopped on both hosts
but its completed artifacts are kept. The launcher waits for all four workers,
then merges their outputs and writes `<run>/pareto/pareto.png`, `pareto.csv`, and
`pareto.json`.

The zero-free quantizer uses symmetric odd levels (`±1, ±3, ...`) with an
MSE-refit scale per row, so small values are never rounded to a dedicated zero
code. The Pareto x-axis is **effective transmitted bits/value**, including
scales, headers, padding, and zlib compression; retained task gain is capped at
the raw-adapter performance.

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
