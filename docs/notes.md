## Prerequisites

Identify a model/dataset combination on which we can greatly and easily improve performance via a (q)LoRA finetune.

This ended up being Mistral-7B on GSM8K. (6% --> 69% exact match).

## Experiment 1

- If we try to compress the LoRA, how does MDL fare against "uniform code" quantization?

see [results/rmse_mdl_lora_vs_quantized_lora/](results/rmse_mdl_lora_vs_quantized_lora/) for the detailed results, but the findings are essentially that MDL is not a good allocator for this problem, and that the uniform code is hard to beat.

MDL drops too many rows by default which crashes its performance against a bit-matched uniform quant, even though it technically has a better RMSE reconstruction error. A no-drop constrait makes it even, but that's it.

## Experiment 2

Let's focus on the dataset compression and informational contents.

Design: 4 arms at fixed compute (2000 steps, 32k samples seen), varying only unique rows — 4k×8ep, 8k×4ep, 16k×2ep, 32k×1ep. Duplication is the axis, not a side control. Measure R*(0.90), the smallest adapter file holding 90% of the gain. If R* tracks information it rises across arms; if it tracks compute it is flat.

Three problems in the way. (1) The uniform ladder only has rungs at 1 and 2 bits and R*(0.90) falls in that gap, so it cannot separate arms differing by under a bit — fix with a blended code giving a deterministic fraction of rows 2 bits and the rest 1, for rates 1.0/1.25/1.5/1.75/2.0. (2) Each GSM8K pass costs ~19 min, too slow for a dense ladder — define R* on held-out bits saved (256 examples, forward-only, seconds) and spend generation on 2-3 accuracy anchors per arm. (3) "Information content" is assumed rather than measured — record zlib bits of the training text (duplication-aware) and base-model code length (duplication-blind); R\* following the first and not the second is the result.

Result: it tracks information. We ended up with 5 arms (a 2k-row arm was added to widen the lever) and R\*(0.90) rises from 0.65 bits at 2k distinct rows to 1.00 at 32k — +0.081 bits per doubling, R² = 0.78, and the extreme arms don't overlap across seeds. The accuracy axis agrees on the ordering. See [results/adapter_bits_track_unique_data/](results/adapter_bits_track_unique_data/).

Fix (1) was not enough on its own: with a 1–2 bit ladder every arm returned exactly 1.02 bits, which was the ladder floor rather than a flat result, so the code now goes below one bit by keeping a fraction of the rank directions and dropping the rest. Fix (3) is still open — we never recorded the zlib/code-length measures, so what we have established is that R*tracks*distinct rows\* at fixed compute, not measured information content.

**adapter description length is set by the size of the behavioural change, not by the information content of the data — task content is a small term on a large constant.**.

## General flow so far

Bit budget of the adapter dictated by the task size (e.g. number) of rows? Affected by it, but not exactly. See initial results on row counting.

Hypothesis: relative informational content of the data wrt the base model's weights is a much better predictor. Currently testing by forcing stylistic changes from the base model, etc.
❯ it's not behavioural change, it's informational load of the dataset wrt to the model, if we see the model as an encoder. And we're trying to show that the less efficient/likely the expression of the dataset wrt the base model's weights is, the more work lora has to do!

Further questionning: where in the layers/LoRA repartition/directions do different types of infromation get stored and are affected? E.g. style of the output may be stored in the later layers, while factual knowledge may be stored in the earlier layers. This could inform how we design our LoRA adapters and which layers we choose to finetune for specific tasks.

Further questionning: how does this impact reasoning, in particular via CoT reasoning?

## Change of objective: functional sensitivity, not corpus information

Three attempts at a corpus statistic that predicts adapter bits have now failed
the same way. Row count, held-out bits saved, and the divergence between base
and adapted output distributions all rank datasets correctly inside one corpus
and backwards across corpora; the divergence turned out to correlate +0.978
with held-out bits saved, so it was never a second measurement at all. What
survives is an invariance: at a fixed rate the weight reconstruction error is
the same for every corpus to about one per cent, so the codec finds no adapter
harder to compress than another. All the variation is in how much behaviour a
given weight error destroys.

The objective is therefore no longer a corpus law. It is a compression
sensitivity law: given a trained LoRA and a perturbation, what predicts the
behavioural damage? Corpus, rank, optimizer budget and data diversity become
factors that set that sensitivity rather than separate stories. Rank and
optimizer depth already produce effects at least as large as corpus at low
rates, so they are first-class axes now.

Two measurement decisions follow. R* is retired as the primary target: retention
curves cross, so no threshold crossing orders the arms. `budget_500` retains
0.470 at 0.21 bits per value and `budget_8000` retains 0.304, while at R*(0.90)
the order reverses, 0.893 against 0.778. The full distortion surface D(c, r) is
the target instead. And the campaign belongs at 0.15-0.35 bits per value, where
the between-study spread is 0.072 against a seed noise of 0.017; near one bit
per value everything is crushed against full retention.

The standing bar for this line of work is an intervention, not a correlation. If
curvature predicts damage but cannot produce a codec that keeps more behaviour
at the same serialized budget, it is not the result.

### The kill test, before any new campaign

81 adapters with roughly seventeen rungs each are already on the cluster, which
is about 1,300 perturbation and damage pairs. For each one the two Taylor terms
around the trained adapter are measured against the damage actually observed:

    dL ~= g.d + 0.5 d'Hd

The gradient term is measured, not assumed away -- these are validation-selected
checkpoints scored on held-out text, so there is no reason for the held-out
gradient to vanish. Order: 2 bits per value first, where the relative
perturbation is smallest, then down the ladder to find where the expansion
breaks. If second order already fails at 2 bits, curvature is not the mechanism
and we do not spend the panel on it.

One caution the numbers already give. On the fp32 tiny-model fixture the
expansion is exact to 0.08% at a 3% perturbation, 3% off at 10%, 27% off at 30%
and useless at 60%. The real codec perturbations are 0.31 relative at 2 bits per
value, 0.55 at 1 bit and 0.93 at 0.21. So even the friendliest rung sits where
the toy expansion is already 25% wrong, and a clean second-order result would be
a surprise rather than the default.
