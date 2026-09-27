# Behavioral rate

The rate depends on the learned update, frozen receiver, task, and codec.
Source length, weight error, and within-receiver correlations do not establish
a predictor that transfers to other receivers.

| Evidence | Role |
|---|---|
| [adapter_bits_track_unique_data](adapter_bits_track_unique_data/) | Distinct content at a fixed training budget; Figure 2 |
| [codec_comparison](codec_comparison/) | Weight reconstruction error can disagree with retained gain; Figure 2 |
| [what_sets_the_adapter_bit_budget](what_sets_the_adapter_bit_budget/) | Diversity, training budget, rank, placement, and curvature controls |
| [transposed_receiver_panel](transposed_receiver_panel/) | Same corpora across receivers; Figure 6 |
| [adapter_spectrum](adapter_spectrum/) | Spectrum and shrinkability predictors; Figure 7 and its audits |
| [rate_law_audit](rate_law_audit/) | Receiver-held-out tests and failed corpus-only rules |
| [correction_spectral_rate](correction_spectral_rate/) | Correction-spectrum development and external checks |
| [correction_channel_rate](correction_channel_rate/) | Locked prospective checks and their failures |

The shrinkability rule has no fitted coefficient in its primary form. Its
42-checkpoint comparison is retrospective. Interpolated crossings, raw-gain
targets, and the first feasible measured file are separate recorded targets;
a low prediction error does not guarantee that the chosen file meets retention.

Other folders retain earlier controls, coverage checks, and negative outcomes.
See the [reproduction guide](../../docs/reproduce.md) for current commands.
