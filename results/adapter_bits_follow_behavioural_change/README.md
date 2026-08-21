# Do adapter bits follow the behavioural change rather than the data?

**Status: pre-registration with motivating evidence. Nothing here is a
finished result.** The figure and table below are a re-analysis of runs made
for a different question; the grid that tests the hypothesis is specified at
the bottom and has not been run.

## The observation

Re-plotting the five Experiment 2 arms with adapter file size against the
behavioural change each coded adapter actually achieves — held-out bits saved
per token — puts all five on one curve.

![adapter file size against behavioural change](collapse.png)

| bits/token achieved | 2,000 rows | 4,000 | 8,000 | 16,000 | 32,000 |
|---|---:|---:|---:|---:|---:|
| 0.20 | 8.80 | 8.80 | 8.80 | 8.80 | 8.80 |
| 0.30 | 16.70 | 16.70 | 11.43 | 16.70 | 16.70 |
| 0.40 | 21.95 | 21.96 | 16.70 | 16.70 | 16.70 |
| 0.50 | 32.47 | 32.47 | 27.21 | 27.21 | 27.21 |

Megabits of adapter file. The arms separate only above about 0.45 bits/token,
where the low-data arms need slightly *more* file for the same change.

## Why that matters

R*(0.90) is defined as the rate retaining ninety percent of that arm's own
gain, so each arm is asked for a different absolute amount of behaviour. Those
targets rise monotonically with unique rows: 0.479, 0.501, 0.527, 0.559, 0.582
bits per token. Along a shared curve, a higher target costs more file. So the
Experiment 2 slope has a second explanation that has nothing to do with
information: more data produced a larger behavioural change, and larger changes
cost more bits.

## Why this design cannot settle it

The two explanations fit the same 15 runs equally well:

| explanation | slope | R² |
|---|---|---|
| R\* against the absolute target (0.9 × own gain) | 2.367 per bit/token | 0.733 |
| R\* against log₂(unique rows) | 0.0618 per doubling | 0.706 |

They are collinear by construction: more unique rows produce a bigger gain
produce a bigger target. No amount of extra seeds separates them. Breaking the
collinearity needs a lever that changes behavioural change while holding the
data fixed.

## The lever

Rewrite what the adapter is taught to emit, leaving the problems and answers
identical. Five deterministic transforms of the response body, with the final
`The answer is: N` line untouched so exact-match scoring is unaffected:

| transform | what changes |
|---|---|
| `plain` | nothing |
| `preamble` | a fixed opening sentence is prepended |
| `numbered` | each line is prefixed `(1)`, `(2)`, … |
| `shouted` | the body is uppercased |
| `symbolic` | fixed lexical substitutions: `is` → `≡`, `the` → `‹the›`, … |

Task information is constant across all five — same problems, same answers,
same rows. What changes is how far the target sits from what the base model
would write, which the run measures as the base model's held-out bits per token
on the transformed responses rather than assuming it from the ordering above.

## Pre-registered grid

Mistral-7B on an NF4 base, rank-16 LoRA, 8,000 MetaMathQA rows, compute-matched
at 2,000 optimizer updates and 32,000 samples seen. Five transforms crossed with
two diversity levels — 400 and roughly 5,620 distinct source problems behind the
same 8,000 rows — at three seeds. 30 runs.

Crossing the two levers is the point. The diversity axis moves information at
fixed behaviour; the transform axis moves behaviour at fixed information.

## Pre-registered predictions

- **P1.** Plotted against achieved behavioural change, all 30 arms fall on one
  curve. Spread between arms at a matched absolute change is smaller than the
  spread between seeds within an arm.
- **P2.** R\*(0.90) rises with the transform's distance and is flat across the
  diversity axis.
- **P3.** Once the absolute target is controlled for, unique-row count adds no
  explanatory power: partial R² below 0.1.

P1 and P2 holding means adapter bits are set by the size of the behavioural
change, and the Experiment 2 result is a consequence of that rather than
evidence about information. P2 failing in the other direction — R\* flat across
transforms and rising across diversity — restores the information reading. A
mixed outcome means both matter and the decomposition is the result.

## Known weakness

The rate ladder has fourteen rungs, so the curve is read at coarse resolution:
several entries in the table above are identical because they are the same rung,
not because the arms agree exactly. The grid should carry a denser ladder
between 0.3 and 0.7 bits per token, where the crossings fall.

## Files

| file | contents |
|---|---|
| `collapse.png` | file size against behavioural change; R\* against each arm's target |
| `rate_against_change.csv` | every (arm, seed, codec) point behind the figure |
