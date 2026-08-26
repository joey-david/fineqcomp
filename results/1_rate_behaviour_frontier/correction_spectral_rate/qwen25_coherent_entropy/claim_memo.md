# Relative-information claim memo

Date: 2026-08-24

This memo is local to `fineQComp`. It does not edit the internship report.

## Bottom line

No single scalar in the tested panel supports a paper-ready, receiver-independent
claim. The strongest positive lead is `coherent_energy_over_entropy`, but its
locked Qwen2.5 test failed. Raw `fisher_logdet` passes the Qwen2.5 panel only in
a posthoc scan and fails the two earlier external receivers. The current claim
must therefore stay exploratory:

> A shared correction-gradient measure can predict the adapter rate on some
> receivers, but the present evidence does not establish a universal
> model-relative information law.

## Locked Qwen2.5 test

The candidate was fixed before opening Qwen2.5 codec results:

\[
I_{\mathrm{coh/H}} = \frac{\lVert \frac{1}{n}\sum_i g_i\rVert_2^2}
 {\frac{1}{n}\sum_i H_M(Y\mid X_i)},
\]

where `g_i` is the per-example output-head correction sketch and entropy is in
bits per token. The row sample was 256 uniform examples (seed 271828); the
sketch used hidden dimension 64, residual dimension 32, and seed 1729.

| receiver | usable natural arms | rho | exact p | candidate RMSE | base-code RMSE | gate |
|---|---:|---:|---:|---:|---:|---|
| Llama-3.1-8B | 5 | 0.70 | 0.075 | 0.1320 | 0.3270 | passed |
| Qwen3-8B | 4 | 0.80 | 0.1667 | 0.1630 | 0.3059 | passed; four-arm screen |
| Qwen2.5-7B | 5 | 0.10 | 0.475 | 0.2958 | 0.2410 | failed |

The Qwen2.5 result is from the locked command and uses all five usable natural
corpora: code, dialogue, instruction, summary, and panel math. All 15 natural
runs have 17 codec files and finite `R*` values. The CoT panel is transfer-only;
one answer arm has no learning and is excluded from `R*` analysis.

## Posthoc candidate scan

The same untouched Qwen2.5 measurements were evaluated for the original ten
measures plus derived and spectral forms. Only raw `fisher_logdet` passed the
Qwen2.5 rho/RMSE gates (rho 0.70, exact p 0.10, RMSE 0.2340 versus base
0.2410). It was not locked for Qwen2.5 and fails the Llama and Qwen3 scans
(rho -0.50 and 0.40; RMSE 0.483 and 0.461). Thus it cannot replace the locked
claim.

`centered_angular_correction_logdet` is positive on all three receivers (rho
0.40, 0.40, 0.60) but misses the rho and RMSE gates. Two-feature exploratory
fits improve RMSE on all three, with `base_codelength_bits_per_token` plus
`margin_deficit_bits` the best simple pair, but no pair passes both gates on
two receivers. These are search results, not confirmatory evidence.

The full posthoc artifacts are under
`qwen25_posthoc_scan/`, `llama_posthoc_scan/`, and `qwen3_posthoc_scan/`.

## Ten original measures

1. Frozen-model target code length per token.
2. Between-example surprisal variance.
3. Target-logit margin deficit.
4. Effective rank of response hidden states.
5. Surprise-weighted hidden-state log-volume.
6. Effective rank of output correction residuals.
7. Fisher trace: mean squared correction-gradient norm.
8. Fisher effective rank.
9. Fisher log-volume (`fisher_logdet`).
10. Linearized NTK cost for one unit of gain.

The implementation also records correction-spectrum rate-distortion/rank and
log-volume forms, plus the derived coherent energy and its entropy-normalized
form. Definitions live in `src/fineqcomp/relative_info.py`.

## Run and validation record

- Qwen2.5 config: `configs/external_qwen25_panel.yaml`.
- Preflight: Jean-Zay job `1310841`, completed successfully.
- Measurement smoke: `1311170_0`, completed successfully.
- Full measurement array: `1311281`, 24/24 completed; one measurement retry
  `1311404_21` completed successfully.
- Full compression array: `1311282`; 23 tasks completed and one transient CUDA
  failure (`_22`). Isolated retry `1311405_22` completed successfully.
- Natural panel: all 15 runs complete with 17 codec files each.
- Local checks: `uv run --frozen pytest -q` passed; targeted Ruff checks passed;
  batch-script syntax checks passed.

The internship report repository was not modified for this run.
