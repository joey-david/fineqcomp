# Recovery across tasks

`arms.csv` records the task battery; `headroom_grid.csv` records the earlier
response-mismatch controls, including summarization, XBRL tagging, and SQL.
These controls test the scope of recovery beyond reasoning. They do not imply
that every task improves above a competent frozen model.

`scripts/analysis/task_battery.py` joins per-example predictions, checks scores,
and computes paired intervals. A **hit** has a corrupted adapter below base
and a tested compression above base; a **repair** improves the corrupted
adapter without clearing base. Choosing the best condition after seeing test
scores makes that interval optimistic: treat such a hit as a lead to confirm.
Missing or failed arms are not zero scores.

The runtime reuses `fineqcomp.studies.generalisation`. Training configs and
fixed-condition sweeps live in `configs/recovery/`; large predictions remain
in the ignored run/report directories. See Appendix D of the
[submission](../../paper/submission.pdf) for the scale and task comparisons
that entered the paper.
