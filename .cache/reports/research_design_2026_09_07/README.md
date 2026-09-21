# Two paper directions: evidence first

**8 September results review:** [New runs, decompositions, and revised priorities](results_review.md). The earlier design recommendations below predate these results.

Research-design pass, 7 September 2026. This is a design and artifact audit, not a report of new GPU experiments.

Read in this order:

1. [Experiment ledgers, measurement repairs, and decision](ledger.md).
2. [fineQComp: 16 ranked directions and three full paper designs](directions.md).
3. [Reasoning trajectories: 16 ranked directions and three full paper designs](../../../reasoning-trajectory-private/experiments/research_design_2026_09_07/README.md).

The original design targeted **prospective receiver-relative correction budgets** for fineQComp. The revised reasoning design targets how ordinary CoT adds useful computation. The linked results review gives the current priorities; the runs do not yet establish a universal information quantity or scaling law.

The alternatives are substantive departures, not a list of extra model and dataset runs. Each direction includes a claim, definitions, a test, a falsifier, a useful negative outcome, related work, a formal target, a compute estimate, and five scores. F16 and S16 explicitly discard the current framing.

Two new local audits accompany the report:

- [Actual passing files versus interpolated budgets](file-budget-audit.json): 80 bracketed saved sweeps; median ratio 1.1264, maximum 2.3046. This reuses recorded evaluations; it does not decode or rescore adapters.
- [Register execution errors and recovery](../../../reasoning-trajectory-private/experiments/research_design_2026_09_07/register-error-audit.json): recomputed local transition correctness from compatible output codes, with endpoint, first-error, repair, and early-stop counts. The toy reliability comparisons are descriptive, not prospective results.

GitHub refs were fetched and inspected without switching branches or changing project code. Local newer artifacts were also read. The local fineQComp checkout is one commit beyond the requested paper branch; the reasoning checkout includes later dependency and computation-interference work. Exact refs and evidence limits appear in the ledger.

No new training jobs were submitted. Direct Jean-Zay access timed out; the configured route uses `lamgate`, which the supplied AGENTS.md restricts. Live queue status and available allocation therefore remain unverified. Compute figures below are planning ranges in aggregate H100-hours, excluding queue time. A 500-hour envelope per project, with an initial decision within 100 hours, is an assumption pending a budget choice.
