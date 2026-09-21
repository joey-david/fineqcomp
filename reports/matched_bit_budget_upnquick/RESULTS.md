# Compression is not the same thing as training small

**Qwen2.5-7B · MetaMathQA rationales permuted across problems · GSM8K · one training
seed (11) · 500 test questions · run on upnquick GPU0, 15–16 September 2026.**

Design and background: [`../matched_bit_budget_v1/README.md`](../matched_bit_budget_v1/README.md).
Figures: `figures_top10/11_matched-bit-budget-grid.png`,
`figures_top10/12_compression-not-low-rank.png`.

## What was already known

Fine-tuning on MetaMathQA rows whose worked explanation was taken from a *different*
problem drives GSM8K from 77.6% to 23.5%. Truncating that damaged rank-16 adapter to
rank 1 and re-coding it at one bit per value restores 83.3% — above the base model it
started from. That result varied precision at a single rank, so two explanations
survived it: the *code* matters (coarse quantization discards what memorised the
mismatched rationales), or the *size* matters (any sufficiently small update would do).

## What this run adds

Rank and precision are commensurable — the file shrinks in proportion to both — so
`rank × bits` names a budget that several different adapters meet. Comparing along
those equal-budget diagonals separates "how many bits" from "how they are spent".

### 1. The baseline reproduced on different hardware

| | Jean-Zay (recorded) | upnquick (this run) |
| --- | ---: | ---: |
| Base model | 77.56% | 74.80% |
| Damaged adapter, uncompressed | 23.50% | 23.00% |
| Rank 1 @ 1 bit | 83.32% | 80.20% |

Within sampling error at 500 rows (±≈4 points), across a different GPU, a different
CUDA build, and a different model-resolution path. The overshoot above base
reproduces: +5.4 points here, +5.8 recorded.

### 2. At equal budget, accuracy is not flat — until the budget gets large

| Budget (rank × bits) | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy spread (points) | 13.8 | **37.6** | **35.0** | **28.4** | 1.2 | 1.4 | 0.4 |

Below ~16 units, *how* the bits are spent dominates, and **in every one of those
diagonals the 1-bit cell wins**. At 32 units and above every arrangement lands on the
damaged adapter's 23%: given enough bits, the damage is reproduced faithfully however
it is encoded. So "only the size matters" is false where the recovery happens, and
trivially true where it does not.

### 3. Low rank from scratch does *not* reproduce the benefit

```
rank-1 LoRA trained from scratch, fp16    20.4%
rank-16 LoRA truncated to rank 1 @ 1 bit  80.2%
base model                                74.8%
```

An adapter trained at rank 1 learns the corruption as thoroughly as one trained at
rank 16 (20.2% vs 23.0%). Restricting capacity during training buys no robustness to
mismatched rationales. This answers the question directly: very low rank at fp16 is
**not** equivalent to a higher-rank compressed adapter.

### 4. The width has to exist during training

Comparing `r1_b1` across arms — *identical* final rank, bit width and file size
(0.40 MB), differing only in the rank the adapter was trained at:

| Trained at | rank 1 | rank 2 | rank 4 | rank 16 |
| --- | ---: | ---: | ---: | ---: |
| Compressed to rank 1 @ 1 bit | **59.8%** | 78.2% | 81.0% | 80.2% |
| As trained (fp16) | 20.2% | 20.8% | 22.2% | 23.0% |

Training wide and then compressing beats training narrow by ~20 points **at the same
final representation**. Compression is not simply selecting a small object; it is
selecting a small object *out of* a larger trained one, and the larger one has to have
existed. Rank 2 is already enough — the effect saturates quickly.

## Where this sits in the literature

The headline phenomenon is **not new**, and the write-up should not imply it is.

- **LASER** (Sharma, Ash & Misra, *The Truth Is In There*, ICLR 2024) replaces weight
  matrices with low-rank approximations and improves accuracy, at times by 20-30
  points, with gains concentrated on data rare in training. The authors describe it
  explicitly as a denoising process that removes erroneous information. That is the
  same shape as result 2.
- **Train Large, Then Compress** (Li, Wallace et al., ICML 2020) and the **lottery
  ticket** line both establish that a small model extracted from a trained large one
  beats the same small model trained from scratch. That is the same shape as result 4.
- **Hooker et al., What Do Compressed Deep Neural Networks Forget?** (2019) shows
  compression disproportionately erases atypical and memorized examples.

What those do not cover, and what is the actual contribution here:

1. **The corruption is injected and controlled.** LASER studies whatever noise
   pretraining happened to contain, observationally. Here the mismatch is a
   deliberate intervention holding the rationale-text marginal fixed, so the thing
   being erased is known rather than inferred.
2. **Rank and precision are separated at matched budget, and precision wins.** LASER
   is pure SVD truncation and cannot see this axis. On this grid, truncation alone
   (`r1_b16`, LASER-like) reaches 39.2% while quantization alone (`r16_b1`, no rank
   reduction at all) reaches 51.0%, and both sit far below the two together
   (`r1_b1`, 80.2%). **At matched budget, coarse quantization beats rank truncation.**
   That refines, and sits in mild tension with, LASER's rank-centric account --
   though in a different setting, on a trained adapter rather than base weights, so
   it is not a direct contradiction.
3. **The training-width dissociation in the adapter setting** (result 4): the "train
   large then compress" result transposed to LoRA rank on corrupted data, with the
   final representation held exactly fixed.

None of this was checked against the literature before the run; the project's own
review covers quantized fine-tuning (LoRAQuant, LQ-LoRA, QA-LoRA, ParetoQ, ApiQ) and
not the denoising and memorization line. A proper related-work pass is owed before
this is written up for any audience outside the project.

## What this does not establish

**The mechanism is still open, and the most dangerous control is absent.** This
project's own earlier work found that scaling a rank-one update to 0.3 recovers most
of binarization's benefit, and concluded that the data did not establish a reliable
advantage over a tuned scale baseline. **This grid contains no norm-matched or
scale-swept control.** Quantization changes an update's magnitude as well as its
resolution, so a plain shrinkage of the damaged update could in principle reproduce
much of the pattern here. Until that control is run, the right claim is the
behavioural one — *compression recovers, and low-rank training does not* — and not
the mechanistic one that compression specifically erases the memorised noise.

Result 4 is the part that shrinkage explains least comfortably: the two sides have the
same rank, bit width and file size, so any shrinkage account has to show their decoded
norms differ by enough to carry 20 points. That is measurable and has not been measured.

**Other limits.** One training seed, where the recorded result used three — seed
variance is the main threat to result 4, which rests on one training run per arm. One
model family, one corruption, one dataset pair. 500 of 1,319 GSM8K questions. Every
cell in the grid was tested, so nothing is selected on the test split, but the
intervals are unadjusted for the number of cells. Arms r8, r32 and r64 were not swept
before the deadline and are recorded as `missing` in `summary.json`.

## Next

1. **The shrinkage control.** Sweep the update scale and add norm-matched rank-one
   comparisons at the decoded norm. This is the experiment that could overturn the
   mechanistic reading, and it needs no new training.
2. **Seeds 22 and 33**, to put result 4 on the same footing as the recorded result.
3. **Ranks 32 and 64**, already configured, to find where the benefit breaks upward.
