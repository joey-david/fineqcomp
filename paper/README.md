# Paper and figures

[submission.pdf](submission.pdf) is the final review-ready PDF supplied on
27 September 2026. Its bytes match the file from Downloads.

`manuscript/` is the local TeX source snapshot that preceded that PDF. It has
an older abstract and draft disclosure sections, so compiling it does **not**
reproduce the final submission. This cleanup preserves its text rather than
guessing the final source from the PDF.

## Reproduce figures and saved scores

From the repo root:

```bash
uv run scripts/reproduce.py
```

This writes rebuilt figures, PNG previews, the breadth table, and four
recomputed predictor reports to `.cache/reproduction/`. It checks every report
against the recorded scores. The layer-location PDF and intro diagram source
are copied and marked as preserved in `summary.json`. Model runs and the final
manuscript are outside this command's scope.

## Rebuild a plot subset

```bash
uv run python paper/plots/make_figures.py --preview .cache/paper-preview
```

PDFs go to `manuscript/figures/`; optional PNG previews go to the requested
directory. `--out DIR` puts `figures/` and `tables/` under another directory.
Pass function names to rebuild a subset, for example
`f5_families f11_rate_rule direction_windows`. Unknown names fail.
Plots read tracked CSV/JSON inputs; they do not need `.cache/` reports, model
weights, or GPU access. Original plot filenames retain the source snapshot's
numbering, which differs from the final PDF.

| Final PDF | Plot function / stored source | Measured input |
|---|---|---|
| Figure 1 | `manuscript/figures/information_intro.tex` | Diagram |
| Figure 2 | `f2_frontier`, `f3_weight_error` | `results/rate/adapter_bits_track_unique_data/`, `what_sets_the_adapter_bit_budget/`, `codec_comparison/` |
| Figure 3 | `f5_families` | `data/recovery_families.json` |
| Figure 4 | `f4_rank_vs_bits` | `data/matched_budget/` |
| Figure 5 | `direction_windows` | `results/recovery/R6*_every_window*.csv` |
| Figure 6 | `f10_predictor` | `results/rate/transposed_receiver_panel/`, `correction_spectral_rate/` |
| Figure 7 | `f11_rate_rule` | `results/rate/adapter_spectrum/attenuation_predictor/` |
| Figure 8 | `a2_negative_battery` | `results/rate/what_sets_the_adapter_bit_budget/` |
| Figure 9 | `a3_rate_law_audit` | `results/rate/rate_law_audit/` |
| Figure 10 | Stored `F4_layer_location.pdf` | `results/rate/budget_matched_rank/`; its original plot generator is absent |
| Figure 11 | `f6_shrinkage` | `results/recovery/denoise_vs_shrinkage/`, `data/matched_budget/` |
| Figure 12 | `direction_windows` | `results/recovery/R6e_every_window_mistral_instruct.csv` |
| Figure 13 | `f9_breadth` | `results/recovery/breadth_panel_summary.json` |

`a4_known_payload` plots Appendix A.5's recorded control, and `a11_table`
rebuilds the full breadth table. `direction_windows` also writes the individual
model panels used by the combined figure.

## Data provenance

`data/recovery_families.json` contains the measured three-seed summaries used
by the existing family-comparison plot. Llama and Qwen values came from
`.cache/reports/research_design_2026_09_07/new_run_analysis.json` (`f03`);
Mistral-Instruct values came from each run's `metrics.json` and
`codec_metrics/sub0_500.json` in `.cache/reports/f5_mistral_instruct/`.
The Mistral base checkpoint remains in the historical reports but is not the
Instruct replication.

`data/matched_budget/` copies only the candidate grid and baseline/raw/coded
metrics consumed by the plots, from
`.cache/reports/matched_bit_budget_jz/qwen25_7b_base/`. Values are unchanged.
The large prediction files and adapter checkpoints remain in their local or
cluster run directories. The original submission and stored layer-location
figure are preserved; the plot command does not rebuild them.
