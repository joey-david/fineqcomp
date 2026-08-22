# Chain of thought under compression

The question this programme exists to answer: when a fine-tune teaches a model
to reason, how much of the adapter is the reasoning and how much is everything
else, and which part survives compression?

Everything here is Mistral-7B-v0.1 on an NF4 base, all-linear LoRA at rank 16,
MetaMathQA with the response split at `The answer is:` into a *reasoning* span
and an *answer* span. Held-out bits saved is measured separately on each span
against the same base model, so the two are directly comparable.

| sub-study | status |
|---|---|
| [`span_supervision/`](span_supervision/) | complete, 9 runs, 3 seeds |

## What is settled

**The two spans cost very different amounts to store.** Inside one adapter
trained on the whole response, the rate needed to keep 90% of the gain is
1.07 bits per value on the reasoning span and 0.58 on the answer span. An
adapter trained on the answer alone needs 0.21. Formatting an answer is cheap;
reasoning is five times dearer.

**Supervision on one span damages the other, badly.** Signed held-out bits
saved against the base model, three seeds:

| trained on | whole response | reasoning span | answer span | GSM8K |
|---|---:|---:|---:|---:|
| answer only | −4.35 | **−4.65** | +2.46 | 0.022 |
| reasoning only | +0.43 | +0.57 | **−2.68** | 0.535 |
| whole response | +0.65 | +0.57 | +2.44 | 0.689 |

Answer-only supervision is worse than not fine-tuning at all: GSM8K falls from
the base model's 0.064 to 0.022, because the adapter has made the reasoning
span 4.6 bits per token *less* likely while memorising the final line to a
held-out NLL of 0.0004.

**Reasoning supervision is unaffected by whether the answer is also
supervised.** The reasoning-span gain is 0.567 with the answer supervised and
0.568 without. The two targets are additive to three decimal places, so
whatever the adapter writes for the answer format is disjoint from what it
writes for the reasoning.

![spans under compression](span_supervision/spans_under_compression.png)

## A measurement bug this exposed

`_behavioral_write` clamped bits saved at zero, so the answer-only arm reported
**0.000** bits saved on the reasoning span when the true figure is −4.65. An
arm that had wrecked the model was indistinguishable from one that had left it
alone, and it sat in the runs directory looking harmless. The clamp is gone and
a test pins the signed behaviour.

## What is not settled

One dataset, one model, one split point. The answer span is 2,110 held-out
tokens against 46,554 for reasoning, so the answer-span figures rest on far
less text. And the split is lexical — everything before `The answer is:` counts
as reasoning, including restatement of the question.

## Files

| file | contents |
|---|---|
| `span_supervision/span_rate_curves.csv` | 135 rows: arm, seed, codec, rate, signed bits saved on each span |
| `span_supervision/spans_under_compression.png` | per-span retention curves, and the damage each target does |
