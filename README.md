# fineQComp

### From reasoning trajectories to information budgets

How many transmitted bits of a learned update are actually needed to preserve
new model behavior? fineQComp measures task gain against the exact serialized
size of LoRA updates.

## Dataset information scaling

This campaign reconnects adapter compression to the original information-
theoretic question: does the amount of information in the finetuning data
predict how many adapter bits are needed to preserve what was learned?

The controlled task keeps prompts, example counts, model, optimizer, and the
16 answer tokens fixed. `random` assigns a fresh random label to every mapping
(4 independent source bits per mapping). `structured_pK` samples only `K`
16-item prototype tables and reuses them across families, so task information
saturates at `64*K` bits while dataset size keeps growing.

Dataset compressibility is measured with a conditional prequential code: the
first block is encoded by the shared base model; every later block is encoded by
a fresh LoRA trained only on the preceding prefix. For each trained prefix we
also run an exact reloadable adaptive-MDL sweep and report the smallest adapter
file retaining 90% of the learned accuracy gain.

On the 2xA40 node (`coktailjet` by default):

```bash
# Cheap sanity check: random vs maximally structured, seed 11.
bash scripts/run_information_scaling_a40.sh pilot

# Four information levels, seeds 11/22/33.
bash scripts/run_information_scaling_a40.sh full
```

Outputs include `prequential.csv`, `adapter_information.csv`, and plots of known
independent task bits against both prequential code length and required adapter
description length. Set `A40_HOST` or `INFO_OUT` to override the defaults.

## Adaptive MDL sweep

The MDL experiment treats the base model, LoRA architecture, and decoder as
shared side information and measures the conditional description length of the
learned adapter. Each LoRA row may be dropped or encoded at 1/2/3/4/8 bits; a
Lagrangian allocator minimizes weight reconstruction error plus a penalty on
code length. The final x-axis is the **actual `.fqmdl` file size**, including the
row allocation map, scales, headers, packing, and zlib compression.

```bash
RUN=runs_literature/<run-id>
bash scripts/run_mdl_distributed.sh "$RUN"
```

The launcher evaluates eight description budgets across the two `upnquick`
A100s and two `ourasi` A6000s, then writes `<run>/mdl/mdl_pareto.png`, CSV, and
JSON. A 0-bit row is genuinely omitted, so its budget can be spent on higher
precision elsewhere. Every `.fqmdl` file is decoded from disk before task
evaluation.

## Fixed-rate Pareto sweep

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
