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

Three problems in the way. (1) The uniform ladder only has rungs at 1 and 2 bits and R*(0.90) falls in that gap, so it cannot separate arms differing by under a bit — fix with a blended code giving a deterministic fraction of rows 2 bits and the rest 1, for rates 1.0/1.25/1.5/1.75/2.0. (2) Each GSM8K pass costs ~19 min, too slow for a dense ladder — define R* on held-out bits saved (256 examples, forward-only, seconds) and spend generation on 2-3 accuracy anchors per arm. (3) "Information content" is assumed rather than measured — record zlib bits of the training text (duplication-aware) and base-model code length (duplication-blind); R* following the first and not the second is the result.

Result: it tracks information. We ended up with 5 arms (a 2k-row arm was added to widen the lever) and R*(0.90) rises from 0.65 bits at 2k distinct rows to 1.00 at 32k — +0.081 bits per doubling, R² = 0.78, and the extreme arms don't overlap across seeds. The accuracy axis agrees on the ordering. See [results/adapter_bits_track_unique_data/](results/adapter_bits_track_unique_data/).

Fix (1) was not enough on its own: with a 1–2 bit ladder every arm returned exactly 1.02 bits, which was the ladder floor rather than a flat result, so the code now goes below one bit by keeping a fraction of the rank directions and dropping the rest. Fix (3) is still open — we never recorded the zlib/code-length measures, so what we have established is that R* tracks *distinct rows* at fixed compute, not measured information content.
