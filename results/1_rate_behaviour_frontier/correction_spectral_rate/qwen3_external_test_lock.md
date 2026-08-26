# Qwen3 external test lock

The first Llama-3.1-8B external test failed the locked Fisher-volume gate.
Posthoc diagnostics identify `fisher_effective_rank` as the next clean
candidate: it is scale-free, has lower Llama absolute prediction error than
base code length, and orders Llama full/reasoning/answer supervision correctly.
This Qwen3-8B base panel is the new untouched receiver test.

- Config: `configs/external_qwen3_panel.yaml`
- Receiver: cached Qwen3-8B-Base, NF4, no chat template
- Measure: `fisher_effective_rank`
- Row contract: 256 uniform rows from the whole staged train split, seed 271828
- Sketch contract: hidden dimension 64, residual dimension 32, seed 1729
- Natural panel: five corpora, rank-16 all-linear LoRA, four epochs, seeds 11/22/33
- CoT panel: full/reasoning/answer targets, one epoch, seeds 11/22/33

The primary gate is natural-arm Spearman rho at least 0.70 and a discovery-
prefit absolute RMSE below the base-code-length fit. CoT reports transfer and
cannot rescue a failed natural gate. No Qwen3 R* result may be used to choose
another candidate.

## Availability note

The receiver screen is part of the result. Qwen3-8B-Base scored 0.8749 exact
match on the 1,319-row panel-math GSM8K test, above the fixed 0.8 saturation
threshold, so all three panel-math cells were screened out before training.
The nine CoT cells were screened out for the same reason. The dependent
analysis therefore requires four available natural arms, records the missing
arms, and does not treat the result as a five-corpus or CoT replication.
