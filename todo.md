# TODO

## Local repair — 10 September 2026

`research/information-budget-search` now gives tied values their mean rank in
`spearman`. Previously a constant measure against an increasing measure returned
1.0; it now returns `None` because correlation is undefined. The 21 focused
prediction/search tests pass, including tied ranks and row-order invariance.
Existing scores from this function need recomputation before use; the effect on
the remote panel has not been measured. No GPU run or deployment occurred in this
repair. The peer-gain helper's docstring now agrees with its output: peer gains
cannot establish that a base already met the task's utility target.

## Receiver-relative information audit — 10 September 2026

The target is the additional information this particular base model needs to
acquire the requested behaviour. Keep that separate from the difficulty of
learning the correction and the cost of encoding it as an adapter.

- [ ] Define the receiver-relative correction quantity and its utility target.
  Separate base-model surprise, achievable likelihood improvement, correction
  information, learning difficulty and encoding cost. Measure the correction
  independently of the final adapter budget so prediction is not circular.
- [x] Test whether transfer, shared and integrated likelihood gains add anything
  beyond the likelihood-change measures that already failed. Reuse receiver-swap
  and response-rewrite controls; do not count closely related scalars as distinct
  information theories. `measure_redundancy` reports the pairwise rank correlation
  among the five measures and, by restricting the search to one measure at a time
  and re-running the nested selection, the honest error each reaches alone. Pairs
  at or above 0.95 are named as indistinguishable and the spread of nested errors
  says whether any measure predicts what the others cannot; measures whose errors
  sit on top of each other are one scalar under five names. `choose` and `develop`
  now take a candidate pool so a restricted search is the same code path as the
  full one, and a fold with nothing eligible is recorded as uncovered rather than
  dropped. The discovery phase prints this beside the screening summary. The
  receiver-swap and response-rewrite controls are not wired in here; they need the
  cell artifacts.
- [ ] Measure the encoding penalty at a common functional target across training
  ranks and compression routes. The roughly 7× gap at matched decision fidelity
  motivates this test but does not establish full functional equivalence. Report
  codebook minima as achieved bounds, not unrestricted necessary information.
- [x] Establish a receiver-relative zero point: distinguish a base that already
  meets the utility target from a fine-tune that failed to learn. Report these
  cases separately from small positive gains and budgets outside the tested grid.
  `no_positive_reference_gain` was one status covering two opposite situations,
  and nothing inside a single cell can separate them. The other receivers on the
  same task can: `zero_point` reports `no_gain_while_peers_gained` when others
  learned and this one did not, `no_gain_and_no_peer_gained` when the task itself
  moved no receiver, `reference_worse_than_base` when the fine-tune hurt, and
  `gained_but_above_grid` separately from all of them, so a coarse grid is never
  read as an absent correction. `load_rows` now carries the base and reference
  bits on excluded cells as well as measured ones, and an artifact written before
  those were recorded returns `unknown_no_anchor` rather than a guess. This is a
  peer comparison, not a utility target: it says which receivers differ, not that
  any base met a standard. Defining that target remains the first open item.
- [x] Put uncertainty on the 90% gain-retention budget, accounting for small
  reference gains and selection among many codecs. Keep calibration selection
  separate from test verification and retain verification failures.
  `sweep` now keeps per-example bits beside every total, and `budget_interval`
  resamples the scored examples to report what a single achieved file cannot: how
  often a budget is reached at all, which codecs could have won, how often a small
  reference gain leaves the target undefined, and how often verification falls
  below the retention it was selected for. Selection and verification resample
  independently. On a synthetic four-codec grid the point estimate can sit at the
  p95 of the resampled distribution rather than its centre — at 50% retention the
  achieved file was 3200 bits while the resampled median was 1600, two codecs
  split the selection 238/162, and verification fell short in 107 of 400
  resamples. `targets.json` now carries `intervals` beside `budgets`.
- [ ] Compare subset-trained probes with actual prefixes of the full learner at
  matched updates. Separate data coverage, repeated exposure and initialization
  effects before interpreting curve extrapolation as prediction along one
  optimization trajectory.
- [x] Audit what each receiver actually trains on: dropped rows, truncated
  responses and supervised tokens. Check the mismatch between training at the
  short context limit and scoring whole responses at the longer limit.
  Every cell trains at 1024 tokens and is scored at 8192, and nothing checked the
  gap: no panel dataset defines an answer marker, so `validate_answer_retention`
  skips all of them, and it is never called from this path anyway. Row loss
  dominates token truncation, and both are receiver-dependent. `kind_summary`
  loses 995 of 8000 rows for Mistral against 692 for Qwen because the prompt
  alone reaches the limit; `kind_code` truncates 641 rows for Mistral against 244
  for Qwen and leaves 4.0% against 2.1% of response tokens unsupervised.
  `text_to_sql` and `xbrl_tags` are untouched. The same corpus is therefore a
  different corpus per receiver, and `full_horizon` — which `curve_forecast`
  extrapolates to — moves with the tokenizer rather than with learning. Decide
  whether to match the two limits before the next full array; that choice changes
  cost and invalidates the queued run, so it is not made here.
- [x] Treat the 36-rule search on 12 discovery cells as screening. Report selection
  uncertainty, calibrated errors and simple baselines; freeze the chosen rule
  before development and outer evaluation. `screening_report` now returns the
  three numbers that were being conflated: the winner's own cross-validated error,
  which is the minimum of a 36-way search and therefore optimistic; the nested
  error, which re-runs the whole selection inside every fold; and the fixed
  baselines, so the margin the search actually bought is explicit. A permutation
  null over the targets bounds how good the best of 36 looks with no relationship
  present, and a leave-one-cell-out pass reports how often the same rule wins at
  all. Checked on synthetic panels of the real 2-family by 6-task shape: with a
  planted law the null p-value is 0.005 and one rule wins every fold; with no
  signal the search still reaches 0.93 while the honest nested error is 1.22, the
  margin over predicting the mean is negative, the null p-value is 0.40 and four
  different rules win. Freezing was already enforced through `--lock`.
- [x] Find a measure that is different in kind, not another scalar in the same
  family. Every one of the five was a likelihood change on the same receiver over
  the same rows, and each needed a fitted law — power, affine or log-affine — to
  become a file size, which is where the fitting freedom and the selection problem
  both live. `codec.spectral_bits` is measured on the update itself and is already
  a size: the singular spectrum of each LoRA pair gives an effective rank, two to
  the entropy of the normalized squared spectrum, and a factorization at that rank
  costs a definite number of parameters. It is what a rank-searching codec is
  actually paying for, so it can meet an achieved budget with a single ratio
  instead of a law. It recovers effective ranks of 1.00, 2.00, 3.79, 6.99 and
  12.76 from updates of true rank 1, 2, 4, 8 and 16, and is unchanged by padding a
  truncated update back to its trained container — the container is not the code.
  It is recorded at every probe checkpoint and for the finished adapter, and it is
  an architecture-and-spectrum estimate that must never be reported as an achieved
  file. On a synthetic panel where the spectrum tracks the budget and the
  likelihood family does not, it separates cleanly: nested error 0.000 against
  0.411 to 0.580, no indistinguishable pairs, selected with null p = 0.005 and one
  winner in every fold. Whether it does that on the real panel is open.
- [ ] After screening, preregister independent ranges of correction complexity,
  sample count, model size and adapter architecture. Require genuinely new tasks
  and a new model family before a general scaling claim; the present panel cannot
  identify those exponents.

Implementation fixes on `research/information-budget-search` through `5872ba6`
(local checks passed; renewed GPU validation remains a separate task):

- [x] Guarantee and check requested probe updates after tokenization drops rows;
  calculate the full training horizon from usable rows and record exposure.
- [x] Make the architecture-only capacity control independent of learned weight
  entropy. Keep actual serialized bytes as the adapter-rate target.
- [x] Check data, helper-code and finished-adapter hashes before reusing cached
  scores in the new runner.
- [ ] Inspect renewed GPU smoke artifacts and the completed prediction outputs
  before treating these fixes as validated in the full campaign. The local smoke
  artifacts are still from the superseded `b84d897e`; the renewed smoke `1976482`
  at `5872ba6` has not been pulled back.

Also on `research/information-budget-search`, at `4745a74`, `f447a26`,
`568bfb3`, `6897381`, `78275f7` and `4daeac7` (308 passed, 1 skipped):

- [x] Record exact untruncated token lengths and the cause of every dropped row
  in `CausalExampleDataset`, so truncation is counted rather than inferred from
  rows that merely sit at the limit.
- [x] Replace the `training_exposure.json` payload with a receiver-exposure
  record that separates dropped rows, cut response tokens and the scored span,
  and surface it through the prediction analysis, keeping the exposure of cells
  that never reached a budget so a failure to learn is distinguishable from a
  corpus the limit truncated.
- [x] Add `screening_report`, its permutation null over the search and its
  leave-one-cell-out winner stability, and print the screening summary from the
  discovery phase so the optimistic and honest errors cannot be read as one
  number.
- [x] Carry per-example bits through `sweep`, add `budget_interval`, and write
  resampled intervals into `targets.json` beside the achieved budgets.
- [x] Give `choose` and `develop` a candidate pool, add `spearman` and
  `measure_redundancy`, and print the redundancy summary from the discovery phase.
- [x] Carry base and reference bits onto excluded cells, add `zero_point`, and
  report it from the discovery phase beside the screening and redundancy summaries.
- [x] Add `codec.spectral_bits`, record it at every probe checkpoint and on the
  finished adapter, and admit it to the panel as a sixth measure. The candidate
  count rises from 36 to 42, which the screening null already prices; the honest
  alternative is to preregister it with the affine bytes-to-bytes law as a single
  hypothesis rather than let it into the search. A measure missing from an older
  artifact now makes its candidates ineligible instead of raising.

Evidence: [round-two review](reports/research_design_2026_09_07/round2_review_2026_09_10.md),
[receiver exposure audit](reports/research_design_2026_09_07/receiver_exposure_audit_2026_09_10.json).
