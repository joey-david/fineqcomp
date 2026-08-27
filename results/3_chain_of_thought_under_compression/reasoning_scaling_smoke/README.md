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
| `diagnostics/` | the RoPE gate traceback and the quarantined pre-gate report |

## Not done here

Stage 3, the new-data smokes over NuminaMath-CoT and OpenR1-Math-220k defined in
`configs/reasoning_data_smoke.yaml`, has not produced evidence yet.
