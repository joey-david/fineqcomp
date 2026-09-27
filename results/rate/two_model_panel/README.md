# The corpus ordering does not survive a change of frozen model

Five corpora, two frozen substrates, three seeds, 30 runs, everything else
held: rank-16 all-linear LoRA, 8,000 rows, four epochs, seventeen rungs.
Job 1273993, complete.

| corpus | Mistral R\* | Qwen R\* | Mistral bits saved | Qwen bits saved |
|---|---:|---:|---:|---:|
| instruction (Alpaca) | 0.643 | 0.774 | 0.618 | 0.181 |
| summarization (XSum) | 0.652 | 0.388 | 0.606 | 0.494 |
| math (MetaMathQA) | 0.737 | 0.411 | 0.596 | 0.092 |
| code (Magicoder) | 0.891 | 0.668 | 0.166 | 0.089 |
| dialogue (hh-rlhf) | 0.913 | 1.176 | 0.289 | 0.225 |

Cheapest to dearest:

    Mistral   instruction < summarization < math < code < dialogue
    Qwen      summarization < math < code < instruction < dialogue

Only dialogue keeps its place. Instruction moves from cheapest on Mistral to
fourth on Qwen; summarization and math each fall by roughly a third in absolute
rate. **The rate a behaviour needs is not a property of the corpus.** It is a
property of the corpus and the frozen model together, which is the outcome the
adapter-capacity literature would predict and the one that makes the negative
in the parent folder a structural claim rather than a null.

## The caveat, and it is large

Qwen2.5-7B is a much stronger base model, so there is far less to learn: bits
saved falls from 0.596 to 0.092 on math and from 0.618 to 0.181 on instruction.
Three of the five Qwen arms therefore estimate R\* from a gain three to six
times smaller than the Mistral arm did, and a small gain makes R\* noisier for
the reasons recorded in `../what_sets_the_adapter_bit_budget/criterion_sweep.csv`.
The reordering could in part be that. What would settle it is matching the arms
on absolute gain rather than on corpus, which needs a per-model row count and
is not what this panel did.

## Files

| file | contents |
|---|---|
| `panel_runs.csv` | 30 runs: study, model, seed, R\*, bits saved, overfit gap |
