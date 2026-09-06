# Fine-tuning selects an algorithm from a model-specific version space

Status: **recorded, and the headline is negative**. 360 adapters trained on
Jean-Zay (jobs 1695308-1695312, about 15 H100-hours). The version-space part of
the claim holds; the model-specific part does not survive contact with the data,
and the prospective targeting rule does no better than picking at random.

## The question

When training data fits several rules, which one does a fine-tune come out
implementing? The claim under test is that it is a whole algorithm, favoured by
the base model before training; that adding more data from inside the ambiguous
region leaves that algorithm alone; and that one example chosen against the base
model's own prior switches it globally.

## The design

Four rule families. Each has four programs that give the same answer on every
input in the training region and four different answers on the probes.

| family | programs | training region | probes |
|---|---|---|---|
| `chain` | apply the rule `steps` times / once / `steps mod 4` times / at most three times | `steps = 1` | `steps` in 4-20 |
| `middle` | second element / middle element / median / second smallest | ascending lists of three | lists of five and seven |
| `count` | count of `a` / its parity / its indicator / vowel count | at most one `a`, no other vowel | two or four `a`, other vowels |
| `field` | the field named `code` / the second field / the longest value / the alphabetically last value | all four cues on one field | all four cues on four fields |

The audit (`--audit`) proves the partition rather than assuming it: every
training-region input gives one answer under all four programs, every probe
gives four, and the training, diagnostic, candidate and probe slices share no
input.

Two stages, in this order.

**Diagnose** never trains. It shows the frozen base model eight in-context
examples from the training region and scores each candidate program's answer to
a held-out separating input by likelihood, which gives a behavioural mass over
the four programs before a single gradient step. The dominant program `D` is the
argmax. The target program `P*` is the runner-up. The targeted example is the
separating input the base is most committed to `D` on, labelled with `P*`'s
answer. All of it is written to `locks/<model>/<family>-seed<n>.json` together
with the predicted post-training program.

**Train** refuses to start without that lock. Ten arms share 128 ambiguous rows
and 256 optimizer updates and differ only in what is added:

| arm | addition |
|---|---|
| `base` | nothing |
| `targeted` | the locked example |
| `matched_difficulty` | one candidate at the same distance outside the training region |
| `matched_loss` | one candidate with the same base-model loss |
| `matched_influence` | one candidate with the same gradient norm at the untrained adapter |
| `random_distinguishing` | one candidate drawn at random |
| `ambiguous_10 … _10000` | that many further rows from inside the training region |

Every single-example arm adds an example that contradicts `D`. The three matched
controls hold constant everything the targeted example could be special for, and
among the candidates that match they take the one the base is *least* committed
to `D` on. What separates the targeted arm from its controls is the diagnostic
and nothing else.

The update budget is held fixed across arms, so the 10,128-row arm buys more
information and not more descent. Epochs are derived per arm and capped.

Every arm is then scored by exact agreement between greedy generation and each
of the four programs on 1,024 probes it has never seen.

## Gates

| gate | statement |
|---|---|
| G1 | every arm reproduces the training region at 0.90 or better on held-out ambiguous inputs |
| G2 | 10 to 10,000 further ambiguous rows move agreement with `D` by at most 0.10 |
| G3 | the targeted example raises agreement with `P*` by at least 0.30 over `base`, and `P*` becomes the argmax program |
| G4 | the targeted gain beats the best control by at least 0.20 |
| G5 | G3 and G4 hold on a majority of seeds for at least one family and model |

G1 failing everywhere means the task is unlearnable at this budget and the study
says nothing. G2 failing means the version space is not stable and the framing is
wrong. G3 passing with G4 failing is the informative negative: the switch would
be an effect of adding any out-of-region example rather than of choosing it
against the model's prior, and the prospective part of the claim would be dead.

Mean accuracy gains without program identification do not support the paper.

## Running it

```bash
sbatch --array=0-0            --export=ALL,MODE=audit    scripts/jean_zay_program_selection.sbatch
sbatch --array=0-2%3          --export=ALL,MODE=diagnose scripts/jean_zay_program_selection.sbatch
sbatch --array=0-0            --export=ALL,MODE=smoke    scripts/jean_zay_program_selection.sbatch
sbatch --array=0-19%20        --export=ALL,MODE=full     scripts/jean_zay_program_selection.sbatch
sbatch --array=0-0            --export=ALL,MODE=report   scripts/jean_zay_program_selection.sbatch
```

Grid: 4 families x 3 base models (Mistral-7B, Qwen2.5-7B, Llama-3.1-8B) x 10
arms x 3 seeds = 360 trained adapters.

## Results

`arms.csv` has one row per trained adapter, `arm_summary.csv` pools them,
`gates.csv` and `replication.csv` hold the decision rule, and `arm_effects.png`
is the one picture worth keeping.

### Gates

| gate | outcome |
|---|---|
| G1 training region fitted | 30/36 cells |
| G2 ambiguous data leaves the program alone | 18/36 cells |
| G3 targeted example switches the program | 4/36 cells |
| G4 no control reproduces the switch | 0/36 cells |
| G5 replication across seeds | none |

G1 fails only on `count`, and only for Qwen2.5 and Llama, which cannot fit
"at most one `a`" at this budget (0.52-0.79 against Mistral's 0.97). Those two
model-family cells are unreadable and everything below survives their removal.

### Fine-tuning does select one whole program

The base arm agrees with a single program on 90% of 1,024 unseen probes on
average (median 0.963), and only 3.8% of its answers match no program at all.
Behaviour after training is program-shaped, not a blend. That much of the
framing is right.

### The selected program is a property of the family, not the model

| family | Mistral-7B | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|---|
| `chain` | `one` | `one` | `one` |
| `middle` | `index_1` | `index_1` | `index_1` |
| `count` | `presence_a` | `presence_a` | `presence_a` |
| `field` | `named_code` | `named_code` | `named_code` |

Nine cells out of nine in every family, three receivers, three seeds, no
exceptions. Each winner is the shortest program in its family: the one that
ignores `steps`, reads a fixed position, returns a bit, or follows the explicit
name.

The pre-training diagnostic *is* model-specific -- on `chain`, Mistral's
likelihood favours `mod4`, Qwen2.5 favours `one`, Llama splits between `mod4`
and `cap3` -- but that preference does not survive fine-tuning. The diagnostic
named the implemented program in 24/36 cells, which sounds like signal until you
notice that a constant per-family guess scores 36/36.

This is what kills the paper's premise. "Different base models should require
different examples" needs them to select different algorithms first, and they do
not.

### More ambiguous data changes nothing, and it is not because it was ignored

| arm | rows | gain on the labelled program | drop on the incumbent | training-region fit |
|---|---|---|---|---|
| base | 128 | -- | -- | 0.955 |
| +10 ambiguous | 138 | +0.013 | -0.017 | 0.962 |
| +100 | 228 | +0.015 | +0.019 | 0.987 |
| +1,000 | 1,128 | +0.016 | -0.028 | 0.9995 |
| +10,000 | 10,128 | -0.008 | -0.030 | 0.9996 |

The extra rows are learned -- training-region accuracy climbs to 0.9996 -- and
they still move the extrapolation nowhere. Ambiguous data carries no information
about which program to pick, and eighty times more of it carries no more.

### One out-of-region example does move the program

| arm | gain on the labelled program | drop on the incumbent | implements incumbent / labelled / third |
|---|---|---|---|
| targeted | +0.088 | +0.141 | 18 / 9 / 9 |
| matched difficulty | +0.103 | +0.129 | 19 / 10 / 7 |
| matched loss | +0.118 | +0.178 | 18 / 10 / 8 |
| matched influence | +0.112 | +0.086 | 20 / 11 / 5 |
| random distinguishing | +0.102 | +0.107 | 21 / 8 / 7 |
| base | -- | -- | 24 / 5 / 7 |

One row out of 129 moves the incumbent's share of cells from 24/36 to about
19/36, against 24/36 for ten thousand ambiguous rows. The lever is real.

### Targeting adds nothing

The five single-example arms are indistinguishable. Targeted minus random
distinguishing is **-0.014** in agreement with the labelled program, winning
17 cells out of 36 -- a coin flip -- and all three matched controls beat the
targeted arm on the mean. The targeted example carried +0.28 more base-model
commitment to the incumbent than the random one, which is exactly the quantity
the selection rule maximises, and it bought nothing.

Where the example lands is also close to unpredictable: 27% of single-example
arms end on the program the example was labelled with, 53% stay on the
incumbent, 20% end on a third program the example never mentioned. Conditional
on moving at all, 57% follow the label against 33% by chance. The label steers
weakly; the choice of input does not steer at all.

### The one model-specific effect is how movable a receiver is

Mean gain on the labelled program from one example, over five arms and three
seeds:

| family | Mistral-7B | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|---|
| `chain` | +0.068 | +0.081 | +0.009 |
| `middle` | +0.071 | +0.165 | +0.119 |
| `count` | +0.014 | +0.004 | -0.026 |
| `field` | +0.073 | **+0.634** | +0.042 |

`field` on Qwen2.5 is the extreme case: agreement with `position_2` goes from
0.10 to between 0.56 and 0.99 on a single row, in all three seeds, while the
same example on Llama and Mistral moves it by under 0.08. Same family, same
incumbent program, fifteen times the displacement. Receivers differ in how far
one example moves them, not in where they start.

## What this closes and what it opens

The decision rule in `paper.md` says to continue only if the targeted example
causes a predicted program-level switch, ambiguous data does not produce the
same change, and the crossover replicates. The second clause holds and the first
does not, so:

- **Experiment 2 (prospective receiver crossover) cannot be run as written.** It
  needs two base models that favour different wrong programs. Across four
  families and three receivers there are none.
- **The prospective claim is not available.** No version of "measure the base
  model, choose the example" is supported: the rule that maximises the
  diagnostic's own quantity performs like random selection.
- **What survives is a different claim.** Fine-tuning on ambiguous data
  implements the shortest program consistent with it, the same one for every
  receiver tested, and no amount of further ambiguous data moves it. One labelled
  counterexample moves it about half the time, to a destination that follows the
  label 57% of the time. That is a description-length statement about SFT, not a
  model-specific one, and it would need a real code length over the program set
  to make "shortest" quantitative rather than eyeballed.
