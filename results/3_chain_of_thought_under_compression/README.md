# Chain of thought under compression

The question this programme exists to answer: when a fine-tune teaches a model
to reason, how much of the adapter is the reasoning and how much is everything
else, and which part survives compression?

The span study uses Mistral-7B and Qwen2.5-7B. The causal trace test adds
Llama-3.1-8B. Every arm uses an NF4 base and all-linear rank-16 LoRA on
MetaMathQA, then scores GSM8K.

| sub-study | status |
|---|---|
| [`span_supervision/`](span_supervision/) | complete, 9 Mistral + 9 Qwen runs, 3 seeds |
| [`conditional_trace_rate/`](conditional_trace_rate/) | both locked primary tests pass across 9 matched pairs; half-bit secondary point incomplete |
| [`reasoning_scaling_battery_lock.json`](reasoning_scaling_battery_lock.json) | prospective native-model, data, GRPO, and scaling-law gates frozen; model and data smokes only |
| [`reasoning_scaling_battery_amendment_1.json`](reasoning_scaling_battery_amendment_1.json) | pre-pilot correction after the original data gate and first L_rel formula failed |
| [`reasoning_scaling_smoke/`](reasoning_scaling_smoke/) | stage 2 passes 8 of 9 models; original stage 3 fails, revised data smoke passes; no micro-pilot yet |

## Locked causal test

The completed span split did not show whether useful reasoning comes from the
trace text itself or from its fit to the problem. The locked test keeps every
prompt, correct final answer, rationale-body multiset, training setting, and
codec fixed, then moves each rationale to an unrelated problem of similar
length. Its target is the lowest exact adapter rate that retains ninety per
cent of the aligned-minus-permuted GSM8K gap. The lock forbids the old
per-run gain denominator and fixed the raw and binary tests before results.
The full prospective contract remains in
[`conditional_trace_rate_lock.json`](conditional_trace_rate_lock.json).

## Result: the problem--trace match carries the gain

Aligned traces beat the same marginal trace text assigned to the wrong
problems by **55.37 GSM8K points** with the raw adapter. The 95% hierarchical
bootstrap interval is 52.75 to 59.46 points. All nine model--seed gaps are
positive, and each model mean is above 53 points.

The gap remains **27.55 points** after binary coding at about 1.02 exact bits
per adapter value, with a 95% interval of 16.73 to 35.52 points. Again, all
nine pairs are positive. Both tests pass every threshold fixed in the lock.

| exact adapter point | mean aligned--permuted gap | share of raw gap |
|---|---:|---:|
| raw BF16 | 55.37 points | 100.0% |
| binary, about 1.02 bpv | 27.55 points | 49.7% |
| blended, about 1.50 bpv | 52.81 points | 95.4% |
| two-bit, about 1.97 bpv | 55.00 points | 99.3% |

The permuted adapters did learn their assigned text: every control passed the
fixed held-out loss gate, and the weakest gain was 0.446 bits per token against
a required 0.02. Their low GSM8K scores therefore do not come from a failed
optimizer run.

This establishes a causal effect for problem--trace assignment on this frozen
MetaMathQA-to-GSM8K, rank-16 LoRA grid. It does not establish that written
reasoning is faithful, that the rate is universal, or that the same number
holds outside these models and tasks.

The next battery widens that claim without treating either one good run or one
bad run as final. It tests Qwen3 thinking models and DeepSeek-R1 distills from
0.6B to 32B, adds newer math traces before code and generated task families,
and defines a fixed-rollout GRPO control that moves reward credit while keeping
the sampled rollouts fixed. Its exact models, rate grid, gates, stop rules, and
held-out scaling-law test are frozen in the linked lock above.

The primary result is complete. The rate-at-90% secondary result is not: six
older aligned Mistral and Qwen runs stored the half-bit adapter but did not
score GSM8K at that point. On the scored grid, seven pairs reach 90% by about
1.51 bpv and two by about 1.97 bpv. Those are upper bounds until the six
evaluation-only cells are filled.

## What replicates on a second frozen model

Qwen2.5-7B, same three supervision targets, same three seeds.

**Answer-only supervision destroys the model, and on Qwen it is spectacular.**
GSM8K falls from the base model's 0.776 to 0.217 — fifty-six points — while
the adapter memorises the final line. On Mistral the same arm went 0.064 to
0.022. Supervising the final answer alone is not merely useless; it is worse
than leaving the model alone, on both substrates.

| model | trained on | reasoning span | answer span | GSM8K (base) |
|---|---|---:|---:|---:|
| Mistral | answer only | −4.65 | +2.46 | 0.022 (0.064) |
| Mistral | reasoning only | +0.57 | −2.68 | 0.535 |
| Mistral | whole response | +0.57 | +2.44 | 0.689 |
| Qwen | answer only | −0.62 | +0.02 | 0.217 (0.776) |
| Qwen | reasoning only | +0.100 | −0.08 | 0.790 |
| Qwen | whole response | +0.101 | +0.02 | 0.832 |

**Supervision targets are additive, to three decimal places, on both models.**
The reasoning-span gain is 0.567 with the answer supervised and 0.568 without
on Mistral; 0.101 and 0.100 on Qwen. Whatever an adapter writes for the answer
format is disjoint from what it writes for the reasoning.

**Each target still damages the span it does not cover**, in the same direction
on both, though far more mildly on Qwen, which already knows both behaviours.

## What does not replicate

The claim this folder previously led with. On Mistral the reasoning span needed
1.07 bits per value to keep ninety per cent of its gain and the answer span
0.58, so reasoning looked five times dearer to store. On Qwen the order
reverses: reasoning 0.471, answer 0.575.

| model | reasoning R\* | answer R\* | answer-only R\* |
|---|---:|---:|---:|
| Mistral | 1.070 | 0.578 | 0.209 |
| Qwen | 0.471 | 0.575 | 0.147 |

The mechanism is visible in the gains: Qwen already reasons well on GSM8K, so
its reasoning adapter installs almost nothing (0.101 bits per token against
Mistral's 0.567) and is correspondingly cheap. **"Reasoning costs more to store
than answer formatting" is a statement about Mistral, not about reasoning.**
This is the same lesson as the two-model panel in programme 1: rate is a
property of the corpus and the frozen model together.

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

The span decomposition still uses one dataset and one lexical split point.
The answer span is 2,110 held-out tokens against 46,554 for reasoning, so its
figures rest on far less text. Everything before `The answer is:` counts as
reasoning, including restatement of the question. The causal result repairs a
different issue: it isolates the problem--trace match, not semantic faithfulness.

## Files

| file | contents |
|---|---|
| `conditional_trace_rate/pairs.csv` | all 9 paired raw and coded accuracy gaps, exact paired rates, and control learning gains |
| `conditional_trace_rate/summary.json` | locked primary decisions, hierarchical intervals, curve means, and the incomplete half-bit cells |
| `span_supervision/span_rate_curves.csv` | 135 rows: arm, seed, codec, rate, signed bits saved on each span |
| `span_supervision/spans_under_compression.png` | per-span retention curves, and the damage each target does |
| `span_supervision/qwen_span_rate_curves.csv` | 120 rows: the same measurement on Qwen2.5-7B |
