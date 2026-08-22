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
