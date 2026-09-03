# Fine-Tuning Selects Algorithms from a Model-Specific Version Space

## Claim

When the training data fits several rules, fine-tuning selects a complete
algorithm favored by the base model. Adding many more ambiguous examples may
leave that algorithm unchanged, while one prospectively chosen example can
cause a global switch in extrapolation.

Different base models should require different examples. Their effective
examples and resulting algorithms should be predictable from behavior measured
before fine-tuning.

## Experiments

### 1. Exact program-selection test

Use small rule families whose competing programs can be enumerated. All
programs agree on the training region and differ on unseen ranges, depths, or
compositions.

Before training, use diagnostic inputs to estimate each model's behavioral mass
over the candidate programs. Freeze one example predicted to remove its
dominant wrong program.

Compare:

- one targeted distinguishing example;
- matched examples for difficulty, loss, and gradient influence;
- one random distinguishing example;
- 10, 100, 1,000, and 10,000 extra ambiguous examples.

Evaluate exact agreement with each program over a large disjoint input set. The
target result is stable extrapolation under more ambiguous data followed by a
discrete, global program switch from the targeted example.

### 2. Prospective receiver crossover

Choose two base models that favor different wrong programs. Lock one example
for each model and the predicted post-training program before any run.

The required crossover is:

- model A changes more under A's example than under B's;
- model B changes more under B's example than under A's;
- both switches match the predicted program.

Compare against random selection, response likelihood, difficulty, influence,
GRAPE, and ReverseGen. Stop if these baselines explain the crossover.

### 3. Executable confirmation

Use code tasks where visible examples or tests permit several implementations
and hidden tests identify the intended behavior. Select one new unit test from
each base model's pre-training candidate programs, then measure its effect over
a large hidden input space.

The natural result must show that one locked, model-specific test changes the
implemented algorithm more than hundreds of ordinary training cases.

## Decision rule

Continue only if the targeted example causes a predicted program-level switch,
ambiguous data does not produce the same change, and the receiver crossover
replicates across seeds. Mean accuracy gains without program identification do
not support the paper.

## Reuse

Reuse the current campaign expansion, adapter training, model loading, seeded
generation, evaluation, and artifact contracts. Adapter compression,
transposed rate-law analysis, and paired GSM correction transfer are out of
scope.
