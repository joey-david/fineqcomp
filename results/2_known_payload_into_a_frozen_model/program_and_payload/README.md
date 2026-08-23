# Known payload bits do not set the adapter's description length either

24 cells, three seeds, job 1276363, complete. Every condition is the same
registry-lookup task on the same prompts with the same optimizer budget; only
the label rule changes, and the source bits are known by construction.

| condition | source bits | R\* bits/value | seed sd |
|---|---:|---:|---:|
| `rule` | 4 | 0.633 | 0.086 |
| `rule` + 1 paid family | 8 | 0.778 | 0.054 |
| + 2 | 12 | 1.112 | 0.519 |
| + 4 | 20 | 0.718 | 0.034 |
| + 8 (every family paid) | 32 | 1.291 | 0.476 |
| `random` | 512 | 1.110 | 0.273 |

Source bits span a factor of 128 and R\* moves from 0.63 to 1.29, non-monotone,
with seed spreads of up to 0.52 on two conditions. `random`, at 512 known
payload bits, needs *fewer* bits per value than `rule` with eight paid families
at 32. Every arm learned the task to about the same degree — 4.44 to 4.56 bits
saved per mapping against a four-bit maximum — so this is not a case of some
conditions failing to learn.

The read: the fixed cost of installing any transformation at all dominates
everything the payload contributes. On constructed data, where the information
content is exact rather than estimated, adapter description length still does
not track it. That is the sharpest form of the negative in
[`../../1_rate_behaviour_frontier/`](../../1_rate_behaviour_frontier/), because
here there is no measurement error in the quantity being tested.

## The denser sweep, and the thing that does move R\*

Seven payload levels rather than five, plus the same sweep under a second
arithmetic rule. 33 cells, job 1285190, complete.

| condition | source bits | R\* | seed sd | gain/mapping |
|---|---:|---:|---:|---:|
| `sum` rule | 4 | 0.633 | 0.086 | 4.50 |
| + 1 family | 8 | 0.778 | 0.054 | 4.50 |
| + 2 | 12 | 1.112 | 0.519 | 4.51 |
| + 3 | 16 | 0.865 | 0.134 | 4.51 |
| + 4 | 20 | 0.718 | 0.034 | 4.44 |
| + 6 | 28 | 0.722 | 0.031 | 4.09 |
| + 8 | 32 | 1.291 | 0.476 | 4.48 |
| **`product` rule** | **4** | **1.457** | — | 3.89 |
| + 2 | 12 | 1.812 | — | 3.49 |
| + 4 | 20 | 0.993 | — | 3.21 |
| + 8 | 32 | 1.417 | 0.022 | 4.49 |

Across seven payload levels the fit is R\* = 0.46 + 0.11 log2(source bits) with
R2 = 0.23 — flat inside the noise. Under the second rule the slope is −0.08 with
R2 = 0.08, the opposite sign. Payload bits do not set the budget.

**The program does.** At four source bits — the same four bits, the same
prompts, the same optimizer budget — `sum` needs 0.633 bits per value and
`product` needs 1.457. Two transformations of identical description length,
2.3 times apart in what it costs to store them. That is the sharpest statement
the campaign has produced about what an adapter is paying for: not how much the
task tells the model, but which map it has to install.

Two caveats on the second rule. Its arms learned less well — 3.2 to 3.9 bits
saved per mapping against 4.5 for `sum` — and a weaker fit can inflate R\*;
and three of its four conditions produced a bracketed crossing in only one
seed, so the level is indicative rather than measured.

## What this does not yet establish

The two conditions with large spreads carry the whole apparent trend, and five
payload levels cannot distinguish "flat" from "a shallow line with noise". A
denser sweep — seven payload levels and the same sweep under a second
arithmetic rule, 33 cells — is queued behind this by the overnight driver
precisely because of that.

`rule_p4` reports two seeds rather than three: one cell's crossing was not
bracketed.

## Files

| file | contents |
|---|---|
| `payload_cells.csv` | 36 rows: study, condition, seed, source bits, R\*, whether bracketed, gain per mapping |
