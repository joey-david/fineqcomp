# The rate–behaviour frontier, and what does not set it

The claim this programme is trying to establish, in the form it survived seven
attempts to break it:

> At fixed architecture and training setup, training-content diversity changes
> the operational rate–behaviour frontier. Across heterogeneous fine-tunes,
> however, the required rate is **not** determined by dataset size, likelihood
> gain, distributional shift, weight reconstruction error, or local curvature.
> Extreme adapter compression is dominated by representation- and
> optimization-dependent functional structure.

| sub-study | what it contributes | status |
|---|---|---|
| [`adapter_bits_track_unique_data/`](adapter_bits_track_unique_data/) | the positive: distinct rows move R\* at fixed compute | complete, 5 arms |
| [`what_sets_the_adapter_bit_budget/`](what_sets_the_adapter_bit_budget/) | the negatives: diversity, transforms, budget, corpora, layers, rank, divergence, curvature | complete, 81 adapters |
| [`behavioural_change_refuted/`](behavioural_change_refuted/) | a pre-registration whose hypothesis the data then refuted | closed, see below |
| `two_model_panel/` | the same five corpora on a second frozen substrate | **running**, job 1273993 |

## The positive

Eight thousand rows in every arm, only the number of distinct MetaMathQA seed
problems behind them changing. R\*(0.90) rises 0.682 → 0.713 → 0.746 across 400,
1,200 and 3,600 distinct problems, six seeds, with the extreme seed ranges not
overlapping. This is the one intervention in the programme: one thing moved,
everything else held.

## The negatives, in the order they were tried

| candidate | result |
|---|---|
| distinct rows, across corpora | ranks correctly inside a corpus, backwards across corpora |
| behavioural change (bits saved) | slope 1.90 inside a 0.16-wide window; 0.12 on transforms, flat on budget |
| optimizer depth | 16× the updates on identical content moves bits saved by 0.04 |
| distributional shift (KL) | correlates **+0.978** with bits saved; it was never a second measurement |
| base-model surprisal on the corpus | r = +0.25 with R\* |
| weight reconstruction error | constant to ~1% across all five corpora at fixed rate |
| local curvature (g·δ, ½δᵀHδ) | second order never adds ranking power; the expansion is invalid below 1 bit/value |

## Why they all failed, as best we can tell

A trained LoRA is a redundant, optimization-dependent representation of a
function, and "how many bits does this behaviour need" is only a well-posed
scalar once a representation and a decoder are fixed. Below one bit per value
the coded adapter is nearly a full parameter-vector norm away from the trained
one, so nothing local can apply. Rank and optimizer depth already move the
frontier as much as corpus does at 0.21 bits per value — the math arms alone
span 0.304 to 0.628 retention there, wider than the whole corpus spread.

## What is running

`configs/model_panel.yaml`, job 1273993: five corpora (MetaMathQA, Magicoder,
XSum, hh-rlhf, Alpaca) × two frozen substrates (Mistral-7B-v0.1, Qwen2.5-7B) ×
three seeds, everything else held at rank 16, 8,000 rows, four epochs, the same
seventeen rungs. Both models are run rather than joining to the existing Mistral
numbers, so the comparison needs no cross-config join and no shared baseline.

The negative is currently one frozen substrate. If the ordering of the five
corpora is preserved on Qwen while the levels shift, the statement is about
corpora; if the ordering changes, it is about the substrate, which is the
result Tan et al. would predict and would be the more interesting outcome.
