# Results

The [submission](../paper/submission.pdf) defines the paper claims. These folders
keep measured tables, controls, and negative results; dated study notes preserve
the order in which hypotheses and tests were chosen. They are not live job status.

| Folder | Question | Paper |
|---|---|---|
| [rate](rate/) | What determines the rate needed to retain a learned behavior? | Sections 3 and 6; Appendices A and C |
| [recovery](recovery/) | Which parts of a corrupted update cause damage, and what repairs it? | Section 5; Appendix D |
| [payload](payload/) | Does known source information determine adapter rate? | Appendix A.5 |
| [tasks](tasks/) | How far do recovery results extend beyond reasoning? | Scale and task controls |

CSV and JSON files hold the evidence. Analysis code lives in `scripts/analysis/`
and paper plots in `paper/plots/`. Large checkpoints, prepared corpora, and
per-example predictions stay in the ignored `.cache/` tree or on the GPU host.
Historical locks retain their original paths and hashes; use a fresh output
directory with the reorganized code rather than changing a recorded lock.
