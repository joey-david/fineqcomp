# Llama external test lock

Recorded after the corrected Mistral/Qwen development analysis and before any
Llama-3.1-8B R* result was opened.

- Config: `configs/external_llama_panel.yaml`
- Receiver: cached base Llama-3.1-8B, NF4, no chat template
- Measure: `dataset_fisher_log_volume`
- Definition: `log det(I + N_unique F_correction)` from a fixed 256-row
  Nyström sample
- Row contract: 256 uniform rows from the whole staged train split, seed 271828
- Sketch contract: hidden dimension 64, residual dimension 32, seed 1729
- Natural panel: five corpora, rank-16 all-linear LoRA, four epochs, seeds 11/22/33
- CoT panel: full/reasoning/answer targets, one epoch, seeds 11/22/33
- R* target: smallest dense-ladder rate retaining 90% of the learned held-out gain

The measure was fixed because it passed every Mistral/Qwen development block:
the pooled block-rank rho was 0.878 with a 10,000-shuffle p below 0.0001, and
the five block correlations were 1.00, 0.50, 0.90, 1.00, and 0.88. A linear
fit on the log-volume itself cut development RMSE to 0.216 from 0.278
model-only and 0.356 base-code-length.

The first stability gate wrongly required invariance to the number of Nyström
rows. A 32-dimensional fix was stable but lost tail structure and failed two
development gates; a multi-row area kept the main rank and RMSE result but
still reversed one CoT block. The final claim fixes the calibration budget at
256 rows. Across 12 arms, three independent row samples give median CV 0.0018
and minimum rank Spearman 0.979; three independent sketches give median CV
0.0020 and identical ranks. This definition was fixed before any Llama R*
result was opened.

The natural-panel primary gate is arm-level Spearman rho at least 0.70. The
absolute prediction uses a linear Mistral/Qwen discovery fit on the log-volume
itself, with no Llama fit, and must beat the same prefit base-code-length
model. CoT is a separate transfer check and cannot rescue a failed natural-
panel gate.

Locked commands:

```sh
python -m fineqcomp relative-information \
  --config configs/external_llama_panel.yaml \
  --manifest prepared/external-llama-manifest.jsonl \
  --run-id RUN_ID --prepared-root prepared \
  --out reports/relative_information_external_llama/RUN_ID.json \
  --rows 256 --row-sampling uniform --row-sample-seed 271828 \
  --hidden-dim 64 --residual-dim 32 --sketch-seed 1729
```

No alternative measure may replace a failure on this panel.
