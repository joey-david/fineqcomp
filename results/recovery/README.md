# Corruption recovery

Section 5 and Appendix D of the [submission](../../paper/submission.pdf)
compare damaged adapters, spectral filters, coding, and scalar shrinkage.
Recovery toward the frozen model and gain above it are separate outcomes.
A weak base or a search-split improvement alone cannot establish a gain.

| Evidence | What it records |
|---|---|
| `rank_bits_grid.csv`, `search_grid.csv` | Rank/precision comparisons on reserved rows |
| `R6*_every_window*.csv` | Contiguous-window searches across model families and sizes |
| `R7*_per_direction_contribution.csv` | Average include/exclude contrasts; not isolated causal effects |
| `fine_test.csv` | Held-out check of a selected filter |
| `breadth_panel.csv`, `breadth_panel_summary.json` | Qwen breadth probes and matched-norm clean control |
| `generalization_gemma.csv`, `generalization_gemma_summary.json` | Gemma transfer check |
| [denoise_vs_shrinkage](denoise_vs_shrinkage/) | Coding, shrinkage, and weight-geometry controls |
| `reports/` | Earlier measured recovery summaries and locks |
| `conditional_trace_rate_lock.json` | Source runs for the matched aligned/permuted control |

Keep matched-norm and full-precision controls when comparing coded updates.
Coarse coding often shrinks an update; its effects are not exactly scalar
rescaling. Window contributions can reflect the other directions that a
contiguous window includes, so they are descriptive contrasts.

Code lives in `fineqcomp.studies.spectral_transfer`,
`fineqcomp.studies.generalisation`, `fineqcomp.studies.matched_budget`, and
`fineqcomp.studies.denoise_vs_shrinkage`. CPU summaries live in
`scripts/analysis/`; the [paper plots](../../paper/README.md) read these tables.
