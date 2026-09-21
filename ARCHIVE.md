# Archive

This branch is the historical record for `fineqcomp`. It exists so that
research directions which did not make it into the thesis
(`../report/tex/03_finetuning_compression.tex`, `04_reasoning_trajectories.tex`,
`05_unification.tex`) or the ICLR draft, and code superseded during cleanup,
stay reachable instead of being deleted outright. `main` is the branch to
build on; this one is for digging something back up.

## What lives only here

- `src/fineqcomp/{information_budget_search,information_budget_prediction,
  policy_complexity,policy_complexity_analysis,policy_data,rule_discovery,
  functional_recoding}.py` and their configs/tests — a line of work asking
  whether adapter rate could be predicted from pre-training features, and two
  further forks (policy-learning complexity, rule-discovery cue exposure)
  built on top of it. Abandoned before producing a result cited in the report.
- `src/fineqcomp/program_selection.py` and `results/4_algorithm_selection/`,
  `runs_program_selection/` — model-specific algorithm selection, an
  unpursued paper direction.
- `results/5_predicting_the_bit_budget/`, `results/6_predictive_scaling/` —
  extensions beyond the three programmes `main`'s README documents.
- The `ultrareview-*` branches (auto-generated code-review scaffolding) were
  deleted without merging here; they were tool artifacts, not research.

## What is shared with main

Everything else: the fixed-checkpoint codec, the three documented programmes,
`compression_beats_baseline/`, and the paired CoT/direct pilot code from
`research/correction-conditioning`, which was merged into `main` directly
rather than only kept here, since the thesis cites it by name.
