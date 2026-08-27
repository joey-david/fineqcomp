# Reasoning scaling battery: model smokes

Stage 2 of the battery frozen in
[`../reasoning_scaling_battery_lock.json`](../reasoning_scaling_battery_lock.json).
Nothing here is a scientific result. The stage asks one question per model: does
it load on a single H100, render its thinking chat template, generate, take a
finite adapter update, and round-trip both codecs? Two training rows per model,
one generated example, two optimizer updates.

Eight of the nine panel models pass. The ninth is blocked by a configuration
gate described below. Total cost was 587 H100-seconds, or 0.163 H100-hours,
against the 4.5-hour stage limit.

| model | trained values | best val NLL | peak GiB | midrise bpv | LoRAQuant bpv |
|---|---:|---:|---:|---:|---:|
| qwen3_0_6b_thinking | 10,092,544 | 0.927 | 1.63 | 3.770 | 1.509 |
| qwen3_1_7b_thinking | 17,432,576 | 0.885 | 3.08 | 3.692 | 1.418 |
| qwen3_8b_thinking | 43,646,976 | 0.700 | 10.19 | 3.632 | 1.343 |
| qwen3_32b_thinking | 134,217,728 | 0.498 | 24.67 | 3.594 | 1.295 |
| r1_distill_qwen_1_5b | 18,464,768 | 0.749 | 3.47 | 3.636 | 1.410 |
| r1_distill_qwen_7b | 40,370,176 | 0.736 | 9.20 | 3.605 | 1.322 |
| r1_distill_llama_8b | 41,943,040 | 1.042 | 9.19 | 3.618 | 1.348 |
| r1_distill_qwen_32b | 134,217,728 | 0.619 | 24.66 | 3.589 | 1.294 |
| r1_0528_qwen3_8b | 43,646,976 | 0.652 | 10.18 | 3.641 | 1.347 |

The last row is quarantined, not passed. Loss values come from two training rows
and mean nothing about the models. The bits-per-value columns are the measured
exact file rate at a 4-bit midrise target and at the LoRAQuant setting, so they
record that both codecs round-trip at every model size, not a rate result.

## One model is gated

DeepSeek-R1-0528-Qwen3-8B ships `rope_scaling.attn_factor = 0.8782`. Transformers
reads `attention_factor`, warns about the unrecognised key, and computes about
`1.1386` from the factor instead. The model therefore runs at a different YaRN
attention scale than its authors specified, and the warning is easy to miss in a
job log. `validate_rope_config` in `src/fineqcomp/modeling.py` now raises before
any GPU work, so job `1438936` failed in 42 seconds by design; its traceback is
in [`diagnostics/`](diagnostics/). The earlier report from this model, taken
before the gate existed, is kept there under a `gated-` prefix and must not be
used as evidence. The cell stays in the panel and needs a parity test against an
official inference implementation before any long-trace run.

## One batch was discarded

The first submission, jobs `1438392`–`1438402`, used fixed temporary filenames
in the preflight path, so nine parallel jobs shared scratch files. All were
cancelled and none of their reports were kept. The fix is commit `b727026`.
About seven H100-minutes were spent before cancellation.

## Files

| file | contents |
|---|---|
| `model_smokes.csv` | one row per panel model, pass status and the numbers above |
| `model_reports/` | the eight passing preflight reports, with environment and codec detail |
| `trace_lengths.csv` | token-length quantiles and the fraction fitting each candidate max_length |
| `data_reports/` | the stage-3 preflight report, measured with the retention gate disabled |
| `diagnostics/` | the RoPE and answer-retention gate tracebacks, and the quarantined pre-gate report |

## Stage 3: the frozen gate failed; the revised smoke passes

The frozen stage-3 condition was 128 train, 64 calibration and 64 test rows per
new source, every trace boundary and answer parsing, and at least 99 per cent
of sampled rows keeping their answer tokens at the planned 4,096-token limit.
It failed. OpenR1-Math retained only 37.5 per cent of sampled final answers and
NuminaMath-CoT retained 96.88 per cent. The failure remains the result under
the original lock.

The pre-pilot revision in
[`../reasoning_scaling_battery_amendment_1.json`](../reasoning_scaling_battery_amendment_1.json)
sets a panel-wide 40,960-token ceiling and states that this panel covers traces
with a checkable final answer. Under that revised scope, both sources retain
100 per cent of sampled answers, and R1-Distill-Qwen-1.5B generated and took
two finite updates on each. Peak memory was 4.18 GiB on Numina rows and 13.04
GiB on the far longer OpenR1 rows.

Both sources failed the gate first, for unrelated reasons, and neither failure
was visible before `validate_answer_retention` measured it.

**OpenR1-Math-220k was truncated.** Its median row is 5,049 tokens and its
longest 17,040, against a 4,096-token training limit, so 80 of 128 rows lost the
closing `</think>` and the boxed answer. Moving the limit to 16,384 failed again
on a fresh sample at 97.66 per cent. The revised limit is 40,960, the smallest
declared context window in the model panel; the R1 distills allow 131,072. The
ceiling itself adds no padding because the collator pads only to the longest
row in a micro batch of one. The long traces still cost more time and memory
than 4,096-token traces, so a measured micro-pilot cost must pass before a grid.

| max_length | NuminaMath-CoT rows that fit | OpenR1-Math rows that fit |
|---:|---:|---:|
| 4,096 | 100.0% | 41.4% |
| 8,192 | 100.0% | 74.2% |
| 12,288 | 100.0% | 89.8% |
| 16,384 | 100.0% | 97.7% |
| 20,480 | 100.0% | 100.0% |
| 40,960 | 100.0% | 100.0% |

**NuminaMath-CoT had no length problem at all** — its longest row is 1,303
tokens. About three per cent of its solutions never write `\boxed{}`, and every
such row sampled, 22 of 768, was an olympiad or AoPS **proof** ending in
`\blacksquare`. A proof has no final answer, so those rows cannot be scored by
exact match and offer no problem-trace boundary for the rationale control to
permute. The panel therefore covers problems with a checkable final answer only.
This is a scope criterion, `answer_bearing_only` in the dataset spec, not data
cleaning: the excluded rows are replaced from a slack pool so the arm keeps its
128 rows and stays compute-matched.

One warning is expected and explained: the R1 distill tokenizers declare
`model_max_length` of 16,384 while their model configs allow 131,072, so rows
above 16,384 log a tokenizer warning. The rotary embedding covers the full
length and the smoke trains and generates normally.

## The first relative-load formula failed its own smoke

Job `1443827` scored 32 MetaMathQA rows with K=8 and returned 8.6318 bits
against a claimed 3-bit ceiling. It retrieved 78.1 per cent of rows, so the
large log loss came from a few confident errors and from summed sequence
likelihood being set by trace length and generic fluency. Its 1.628-bit Fano
field also measured information the frozen decoder already recovered, not a
lower bound on adapter size. This job is a method diagnostic, not evidence for
the scaling claim.

The amended schema scores trace-body tokens only, subtracts each trace's
log-mean likelihood across its matched prompts, caps the primary load at the
uniform K-way loss, and reports the raw score, accuracy, Fano directions, and a
random-assignment check separately. It calls this a fixed predictor candidate,
not a mutual-information estimate. No panel measurement can start until the
same 32-row cell and a second frozen sample pass that schema.

## Not done here

Stage 2 and the revised stage-3 run are static and smoke evidence only. No
micro-pilot has run, the GRPO fixed-rollout gate has not been exercised on a
GPU, and no valid relative-load value exists yet.
