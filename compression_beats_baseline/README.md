# Compression beats the baseline

Everything about one result lives here: **an adapter trained on corrupted
reasoning traces, compressed hard enough, scores above the model it was trained
from.** This folder is the index and the narrative; the raw run outputs stay in
`../results/` and `../reports/` where the code writes them, and every section
below links to them.

## The setup, for someone who has not seen this codebase

Take Qwen2.5-7B-Instruct, freeze it at NF4, and attach a rank-16 LoRA adapter to
every linear layer. Train it on MetaMathQA — a maths dataset where each problem
comes with a worked explanation and a final answer — but **first swap the
explanations between problems**. Every row now shows a problem, a worked
explanation belonging to a *different* problem, and the original problem's
answer. Then score GSM8K (500 held-out questions, exact match on the final
number).

| | GSM8K |
| --- | ---: |
| frozen base model, no adapter | 74.8% |
| after fine-tuning on the corrupted data (rank 16, fp16) | 23.0% |
| the same adapter truncated to rank 1 and re-coded at 1 bit/value | **80.2%** |

The fine-tune destroys the model. Throwing away all but one singular direction
of that damaged update and rounding every remaining value to one of two levels
does not merely repair it — it lands **above the base model the run started
from**. Replicated on Mistral-7B and Llama-3.1-8B (+13 to +59 points over the
uncompressed fine-tune in every family).

Compression here means two separate knobs: **rank** (keep the top *k* singular
directions of the update, via a balanced SVD) and **precision** (a uniform
midrise quantizer that writes the surviving values to a reloadable `.fqcb`
file at *b* bits each). Both are applied after training; no retraining happens.

## What is established, in the order the controls were run

### 1. The effect itself
Jean-Zay, 8 September, three model families.
Figures [`02`](figures/02_f03-compression-recovery.png) and
[`06`](figures/06_f03-prefix-collapse.png) · report
[`../reports/research_design_2026_09_07/`](../reports/research_design_2026_09_07/)

The damaged fine-tune does not just lose accuracy, it **collapses onto one
rationale**: 58–73% of its answers open with the same 24-word off-topic
explanation. After compression that share is 0.1%.

### 2. Bits spent at rank ≠ bits spent at precision
upnquick GPU0, 15–16 September.
Figures [`11`](figures/11_matched-bit-budget-grid.png) and
[`12`](figures/12_compression-not-low-rank.png) · report
[`../reports/matched_bit_budget_upnquick/RESULTS.md`](../reports/matched_bit_budget_upnquick/RESULTS.md)

At equal `rank × bits`, accuracy spreads by 37.6 / 35.0 / 28.4 points at budgets
4 / 8 / 16, then flattens to ≤1.4 points at 32 and above. The 1-bit cell wins
every low-budget diagonal.

Crucially this is **not** a capacity-restriction effect. A rank-1 LoRA trained
from scratch on the same corrupted corpus scores 20.4% — it learns the
corruption as thoroughly as rank 16 does. At identical final rank, bit width and
file size (rank 1 @ 1 bit, 0.40 MB), an adapter *trained* at rank 16 reaches
80.2% and one trained at rank 1 reaches 59.8%. **The width has to have existed
during training**; compression then selects a small object out of a large one.

### 3. The recovery is shrinkage — a negative result
upnquick GPU0, 16 September.
Figures [`14`](figures/14_shrinkage-control.png),
[`18`](figures/18_recovery-landscape.png),
[`19`](figures/19_shrinkage-no-free-lunch.png) · study
[`../results/3_chain_of_thought_under_compression/denoise_vs_shrinkage/`](../results/3_chain_of_thought_under_compression/denoise_vs_shrinkage/)

Quantization coarsens an update *and* shrinks it. Simply rescaling the same
rank-one update by α = 0.5 at fp16 scores **80.8%** — against the coded cell's
80.2%, a difference of −0.6 points, 95% CI [−2.8, +1.6]. Against the best point
on the α curve the code has **no advantage at all**.

So the honest headline is that the recovery is a **shrinkage** effect, and any
claim that "compression repairs a damaged adapter" owes the reader this scalar
baseline. This refutes the reading the matched-budget grid left standing.

### 4. What survives shrinkage: a small, real coding effect
Figures [`20`](figures/20_denoising-efficiency.png),
[`22`](figures/22_corruption-intervals.png)

Compared against a rescaling of the *same* directions to the *same* update norm
(matched to a ratio of 1.000000, so the only difference is having gone through
the codec), the coded cell wins at 1, 2, 4 and 8 bits — +2.4, +7.2, +3.0, +1.8
points, three of four excluding zero. The gap in *corruption preference*
(reasoning-span NLL on paired permuted/aligned rows) grows as the code gets
coarser and vanishes where the code is lossless. Per-row likelihoods show the
coded update beating a rescaling carrying 21% more magnitude: −0.0210 bits/token
[−0.0232, −0.0188].

### 5. It does not rediscover the clean solution
Figure [`15`](figures/15_distance-to-clean-adapter.png)

An adapter trained on *correct* rationales scores 80.4% — statistically the same
place. But the compressed corrupted update is near-orthogonal to it: cosine
**0.006**, where the clean adapter's own rank-1/1-bit compression sits at 0.343.
Two different objects that land at the same accuracy.

### 6. What the surviving direction actually does
upnquick GPU0, 16–17 September.
Figures [`16`](figures/16_cot-length-is-not-the-mechanism.png),
[`17`](figures/17_step-boundary-selectivity.png),
[`21`](figures/21_form-versus-benefit.png),
[`23`](figures/23_what-the-direction-reads.png) · study
[`../results/3_chain_of_thought_under_compression/cot_verbosity_mechanism/`](../results/3_chain_of_thought_under_compression/cot_verbosity_mechanism/)

The compressed adapter writes 45% more words before the answer. Three
independent controls close that explanation:

- **Not length.** Forcing the base model to the same chain length (marker bias,
  or a hard floor on finishing) costs accuracy: at 84.9 forced words it scores
  69.4% against the adapter's 80.2% at 88.0 words — a gap of **+10.8 points
  [+6.6, +15.0]** with length held equal.
- **Not a stopping prior.** The update shifts the decision to stop by 0.13 nats,
  two orders of magnitude too small to produce the observed change.
- **Not marginal token preference.** Copying the update's average per-token
  effect (top-80 tokens) reproduces 37% of the surface form and 7% of the
  benefit.

What it does instead: it moves probability mass from `.` to `.\n` — two thirds
of everything it moves — and applies it **conditionally**, 93% sentence
segmentation at 0.8% decimal damage, where a context-blind bias of matched
strength gives 96.2% segmentation at 22.6% decimal damage. An activation probe
on the rank-one direction shows it reads step boundaries at **10.9× sampling
noise in 131 of 196 modules**. The effect is redundant across the lower half of
the stack.

### 7. The gain above base is a format prior — a retracted claim, kept visible
Section 6 of the mechanism write-up originally concluded the effect was
"genuinely weight-level, not reducible to an in-context prior", on the strength
of an experiment where MetaMath-register exemplars lost 4.2 points. That
experiment was confounded: the exemplar sets differed in problem *distribution*
as well as register. The matched retest reversed it — **three in-context
exemplars in the adapter's own segmented register reproduce the full +5.4
[+1.8, +9.1]**. The retraction is preserved in the write-up rather than edited
away.

### 8. Where the damage lives: the dominant directions
Figure [`24`](figures/24_spectral-bands.png)

Slicing the update into contiguous bands of singular directions:

- the **top 4 directions alone**, at fp16, reproduce the damaged adapter (23.4%
  against 23.0%) — the corruption is concentrated there;
- the **tail alone**, at fp16, scores **82.8%, +8.0 [+4.2, +12.0]** — the best
  number in the project, with no compression involved at all;
- compressing *helps* every band that contains dominant directions (+34 to +48)
  and *hurts* the clean tail (−6.8 [−10.2, −3.4]).

This inverts the intuition that compression removes high-frequency noise: the
corruption is the low-frequency, high-energy part, and coarse coding's benefit
is that it damages that part faster than it damages the rest.

## Where everything is

| what | path |
| --- | --- |
| this index | `compression_beats_baseline/README.md` |
| running work log, with the tutor's questions and the answers | [`worklog.md`](worklog.md) |
| ICLR abstract draft (cross-project; this result is its headline) | [`iclr_abstract_draft.md`](iclr_abstract_draft.md) |
| all figures for this result, PNG + light/dark SVG | [`figures/`](figures/) |
| standalone HTML write-up of the recovery study | [`report_recovery.html`](report_recovery.html) |
| published version of that write-up | https://claude.ai/code/artifact/8f462f64-405c-4d8c-b40e-a86ea113eaa4 |
| shrinkage / denoising study | [`../results/3_chain_of_thought_under_compression/denoise_vs_shrinkage/`](../results/3_chain_of_thought_under_compression/denoise_vs_shrinkage/) |
| mechanism study | [`../results/3_chain_of_thought_under_compression/cot_verbosity_mechanism/`](../results/3_chain_of_thought_under_compression/cot_verbosity_mechanism/) |
| matched bit budget grid | [`../reports/matched_bit_budget_upnquick/`](../reports/matched_bit_budget_upnquick/) |
| original three-family design | [`../reports/research_design_2026_09_07/`](../reports/research_design_2026_09_07/) |
| paired CoT/direct pilot | [`../reports/cot_compression_2026_09_10/`](../reports/cot_compression_2026_09_10/) |

Code, all under `../src/fineqcomp/`: `denoise_vs_shrinkage.py` (shrinkage and
norm-matched controls, update geometry), `cot_verbosity_mechanism.py` (length,
termination, readout, layer bands, spectral bands, activation probe),
`matched_budget.py` (the rank × precision grid), `generalisation.py` (the
task-agnostic sweep), `data.py` (the corruption transforms). Configs under
`../configs/upnquick/`.

The figures numbered 02, 06, 08, 11 and 12 are copies; the originals stay in
`../../figures_top10/`, which indexes the strongest results across three
projects and whose numbering other documents already link to.

## What is not established

- **One model, one seed** for everything from section 3 onward. The three-family
  replication covers only section 1.
- **Why shrinkage works at all** is not answered — only that it does, and that
  coarse coding adds a small amount on top of it at matched norm.
- Whether any of this **generalises past chain of thought**. That is the open
  question below.

## Open: does it generalise?

Designed and validated locally, not yet run. Five arms, each training an adapter
on a procedurally corrupted version of a *retrieved* dataset (no synthetic data),
then sweeping the same compression conditions:

| arm | task | corruption | prediction |
| --- | --- | --- | --- |
| `cot_permuted` | GSM8K | rationale from another problem (reference) | recovery + overshoot |
| `cot_arithmetic` | GSM8K | wrong working, answer moved to match it | recovery + overshoot, if the format prior is corruption-independent |
| `cot_shuffled` | GSM8K | reasoning steps reordered | little damage, so little to recover |
| `cls_flipped` | PAWS | labels moved to the wrong class | **recovery but no overshoot** — there is no chain to segment |
| `summary_mismatched` | XSum | summaries dealt to the wrong articles | **overshoot if the effect is generative format rather than reasoning** |

Design: `../configs/upnquick/generalisation.yaml`. Driver:
`../scripts/upnquick_generalisation.sh`. Corruption transforms and their tests:
`../src/fineqcomp/data.py`, `../tests/test_corruption_controls.py`.

## Instrument defects found and corrected in this line of work

Kept on the record because two of the five reported a null that was not there,
which is the reason to treat any single un-cross-checked measurement as
provisional:

1. Results were stratified against the wrong copy of the test corpus (3/500
   items matched); a "+17.6 points on the 9+ stratum" claim was retracted and
   re-derived as +6.1 / +5.5, roughly uniform.
2. Few-shot exemplars were truncated before their answer line — they
   demonstrated a register but not how to finish in it.
3. A base model given exemplars invented further `Question:` blocks in 96% of
   responses, and the "last number in the response" scorer read the fabrication.
   Fixed with a stop-cut applied before any metric is taken.
4. Log-probabilities were averaged where probabilities were needed, making a
   token readout return a geometric mean of ≈ e⁻¹⁵ for every token in the
   vocabulary — a spurious null.
5. An activation coefficient was measured in absolute value where the sign
   carries the meaning, reporting a flat ratio of 0.988 — a second spurious
   null. The signed measure shows 10.9× sampling noise.
