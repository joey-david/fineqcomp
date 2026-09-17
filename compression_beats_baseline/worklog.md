# Work log: compression of a corrupt LoRA

## Compression of a corrupt LoRA

(_trained on mismatched reasoning/result pairs, GSM8K_) increases performance over the base finetune and over the base model)
Original experiment is [here](../reports/research_design_2026_09_07/).

1. Checked that "full finetune" did refer to an uncompressed LoRA rather than a full model finetune (it does).

2. **Matched bit budget: done.** Run on upnquick GPU0 overnight 15–16 Sept. Full
   write-up: [`../reports/matched_bit_budget_upnquick/RESULTS.md`](../reports/matched_bit_budget_upnquick/RESULTS.md).
   Figures: `figures/11_matched-bit-budget-grid.png`, `figures/12_compression-not-low-rank.png`.

   The rank × precision grid was swept on the damaged rank-16 adapter, and low-rank
   adapters were trained from scratch on the same permuted corpus for comparison.
   Qwen2.5-7B, seed 11, 500 GSM8K questions. The baseline reproduced the Jean-Zay
   numbers: base 74.8 vs 77.56, damaged 23.0 vs 23.50, rank-1/1-bit 80.2 vs 83.32.

   - **At equal `rank × bits`, accuracy is not flat.** Spread is 37.6, 35.0 and 28.4
     points at budgets 4, 8 and 16, and collapses to 1.2, 1.4 and 0.4 at 32, 64 and
     128. How the bits are spent dominates in the compression regime and stops
     mattering once there are enough bits to reproduce the damage. **The 1-bit cell
     wins every low-budget diagonal.**
   - **Low rank from scratch does not reproduce the benefit.** A rank-1 LoRA trained
     at fp16 scores **20.4%** — it learns the corruption as thoroughly as rank 16
     (23.0%). It is *not* equivalent to a compressed higher-rank adapter (80.2%).
     Restricting capacity during training buys no mismatch robustness.
   - **The width has to exist during training.** At *identical* final rank, bit width
     and file size (rank 1 @ 1 bit, 0.40 MB), an adapter trained at rank 16 reaches
     80.2% while one trained at rank 1 reaches **59.8%**. Rank 2 already recovers
     (78.2%), so the effect saturates fast. Compression selects a small object *out
     of* a larger trained one, and the larger one has to have existed.

   **Caveat that matters:** no norm-matched or scale-swept control was run, and this
   project previously found that scaling a rank-one update to 0.3 recovers most of
   binarization's benefit. So the behavioural claim is solid — compression recovers,
   low-rank training does not — but "compression specifically erases the noise" is
   not yet established against plain shrinkage. Also one seed, one model.

   **Next:** (a) the shrinkage/norm-matched control, which needs no new training and
   could overturn the mechanistic reading; (b) seeds 22 and 33; (c) ranks 32 and 64,
   already configured, to find where the benefit breaks upward.

- TODO: ~~Check at matched bit budget (fixed rank\*precision) and see dynamics. Try an array of rank/precision combinations - check if LoRA trained at lower ranks from scratch have the save beneficial effect. Try very low rank at fp16 and see if it's equivalent in mismatch robustness to a higher rank compressed LoRA.~~ Done — see above.

Tutor: "Ouais en gros si quand tu finetunes un rang plus faible mais avec une meilleure précision, vu que t'as moins de poids mais qui coûtent plus chers chacun, tu peux avoir le même nombre de bits à rang différent et donc des comportements différents alors que tu bouges autant de bits

Ce serait bien du coup de faire ton xp de finetuning sur un dataset modifié en faisant varier les rangs pour voir si ce que tu observes sur un rang (la compression permet d'ignorer le bruit et donc de pas sur apprendre un mauvais dataset) est valide de manière générale ou si ça casse quelque part"

→ **Answer:** the same number of bits at different ranks does give different
behaviour, and by a lot (up to 37.6 points). It breaks in two places: upward, once
the budget is large enough to reproduce the damage (≥32 units, spread ≈ 0), and
downward, when the adapter was never wide during training (59.8% instead of 80.2%).
