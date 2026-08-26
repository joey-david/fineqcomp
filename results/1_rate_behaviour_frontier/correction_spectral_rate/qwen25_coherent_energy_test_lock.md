# Qwen2.5 external test lock

This is a fresh receiver test for a measure fixed before any Qwen2.5 R* result
is opened. The Qwen3 effective-rank test failed its natural-arm gate. The new
measure is derived from the correction sketch rather than selected from the
Qwen2.5 outcomes.

- Config: `configs/external_qwen25_panel.yaml`
- Receiver: cached Qwen2.5-7B base, NF4, no chat template
- Measure: `coherent_energy_over_entropy`
- Definition: `||mean_i g_i||^2 / mean_i H_M(Y|X_i)`, where `g_i` is the
  per-example output-head correction sketch and entropy is in bits per token
- Row contract: 256 uniform rows from the whole staged train split, seed 271828
- Sketch contract: hidden dimension 64, residual dimension 32, seed 1729
- Natural panel: five corpora, rank-16 all-linear LoRA, four epochs, seeds 11/22/33
- CoT panel: full/reasoning/answer targets, one epoch, seeds 11/22/33

The primary gate is natural-arm Spearman rho at least 0.70 and discovery-prefit
absolute RMSE below the base-code-length fit. CoT reports transfer and cannot
rescue a failed natural gate. If the receiver screen removes arms, the report
must state the available-arm count and must not claim a five-corpus replication.

The measure has a direct identity: `||mean_i g_i||^2` equals Fisher trace times
mean gradient coherence. Dividing by frozen predictive entropy removes a
receiver uncertainty scale while retaining the shared correction direction.
