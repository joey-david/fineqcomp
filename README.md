# fineQComp

### How many bits does a fine-tune actually need to write?

After a frozen model learns a task through a LoRA, how many transmitted bits are
needed to keep the behavior it gained? fineQComp treats the base model, adapter
architecture, and decoder as shared side information, then measures the exact
serialized size of the learned update against the task gain it retains.

Every rate we report is the size of a real file that we decode from disk before
evaluating it. Nothing is estimated from a nominal bit width.

## The three experiments

| Experiment | Question | Entry point |
|---|---|---|
| Adaptive MDL frontier | What is the shortest adapter description that keeps the behavior? | `fineqcomp.mdl` |
| Fixed-rate controls | How do uniform and LoRAQuant codes compare at matched file rates? | `fineqcomp.pareto` |
| Dataset information scaling | Does task information predict the adapter bits needed? | `fineqcomp.information_scaling` |

The MDL frontier is the headline. The fixed-rate sweep is its control arm, and
the information-scaling study is the controlled calibration that ties adapter
bits back to how much there was to learn.

## Adaptive MDL sweep

Each LoRA row may be dropped or encoded at 1/2/3/4/8 bits. A Lagrangian
allocator minimizes weight reconstruction error plus a penalty on code length,
and the x-axis is the **actual `.fqmdl` file size**, including the row selector
map, scales, headers, packing, and zlib compression. A 0-bit row is genuinely
omitted, so its budget can be spent at higher precision elsewhere.

```bash
RUN=runs/<run-id>
bash scripts/run_mdl_distributed.sh "$RUN"
```

The launcher builds the shared row rate/error table once, then evaluates eight
budgets across the available GPUs and writes `<run>/mdl/mdl_pareto.png`, CSV,
and JSON.

## Fixed-rate controls

Reuse an already-trained run rather than training again:

```bash
RUN=runs/<run-id>
bash scripts/run_pareto_distributed.sh "$RUN"
```

## The quantizer, and why it matters

All three experiments share one quantizer, `codec.midrise_quantize`. It uses
zero-free symmetric levels (`±1, ±3, ...`) with a least-squares row scale, so no
codeword is spent on an exact zero and every codeword is used. At one bit it
reduces to `sign(w) * mean(abs(w))` per row.

The campaign also stores `midtread2` and `midtread3`: the usual absmax quantizer
with an exact zero level, at a matched payload rate. It is a control, not a
baseline. At two bits it represents only `-1, 0, +1`, which costs roughly 2.5x
the reconstruction error of the mid-rise code at the same packed width, and
leaves it no better than a one-bit sign code at a comparable file rate. Keeping
both on the same axes is how the campaign separates the effect of code geometry
from the effect of rate.

## Dataset information scaling

The controlled task holds prompts, example counts, model, optimizer, and the 16
answer tokens fixed. `random` assigns a fresh random label to every mapping (4
independent source bits each). `structured_pK` samples only `K` 16-item
prototype tables and reuses them, so task information saturates at `64*K` bits
while dataset size keeps growing.

Dataset compressibility uses a conditional prequential code: the first block is
encoded by the shared base model, and every later block by a fresh LoRA trained
only on the preceding prefix. For each trained prefix we also run the MDL sweep
and report the smallest adapter file retaining 90% of the learned accuracy gain.

```bash
bash scripts/run_information_scaling_a40.sh pilot   # random vs structured, seed 11
bash scripts/run_information_scaling_a40.sh full    # four levels, seeds 11/22/33
```

## Full campaign

```bash
./scripts/bootstrap.sh
.venv/bin/python -m fineqcomp prepare
.venv/bin/python -m fineqcomp run --shard 0 --shards 2
.venv/bin/python -m fineqcomp analyze
```

The campaign trains rank-16 adapters on MetaMathQA, Magicoder, and XSum with
Mistral-7B and Qwen2.5-7B, then evaluates GSM8K, MATH, HumanEval, and XSum. Runs
are restart-safe and preserve the raw adapter, so every codec point is derived
from one trained checkpoint and training variance cannot masquerade as a
compression effect.

See `experiments.md` for the protocol and claim gates, and `literature_review.md`
for how the design follows from prior work.
