# Positive results

Everything that held up, from the discovery that compressing a corrupted adapter
beats the base model (8 September 2026) to the scale and task grids
(18 September). Nulls, refutations and cells that cannot support a claim are in
`README.md`, not here.

## 1. The core effect

- Qwen2.5-7B, rank-16 LoRA on MetaMathQA with each rationale swapped to a
  different problem: GSM8K **74.8% → 23.0%**. Truncated to rank 1 and coded at
  1 bit: **80.2%** — above the base model it was damaged from.
- Replicates in three families: Mistral-7B 6 → 16 → 29, Llama-3.1-8B
  10 → 20 → 42, Qwen2.5-7B 78 → 24 → 83.
- Independently reproduced on Jean-Zay from a fresh training run on different
  hardware: **75.6 / 22.8 / 80.8** against the recorded 74.8 / 23.0 / 80.2.
- The damaged fine-tune **collapses onto one rationale**: 58–73% of its answers
  open with the same 24-word off-topic explanation. After compression, **0.1%**.

## 2. The problem–trace match is what carries the gain

- Aligned rationales beat the same text assigned to the wrong problems by
  **55.37 GSM8K points**, 95% hierarchical bootstrap [52.75, 59.46]. All nine
  model–seed gaps positive; every model mean above 53.
- The gap survives compression: **27.55 points** after binary coding at ~1.02
  exact bits/value [16.73, 35.52], again all nine pairs positive.
- Retention by code: two-bit (~1.97 bpv) **99.3%** of the raw gap, blended
  (~1.50 bpv) **95.4%**, binary (~1.02 bpv) 49.7%.
- The permuted adapters genuinely learned their assigned text — weakest
  held-out gain 0.446 bits/token against a 0.02 gate — so the low scores are not
  a failed optimiser run.
- On **fresh numeric instances** (GSM-Symbolic), aligned beats permuted by
  **54–57 points** on Llama across three seeds, so the gap is not memorisation
  of the test set.

## 3. Compression preserves a *good* adapter too

- Correctly paired CoT training, three seeds, raw → 2-bit → 1.5-bit → binary:
  Mistral **68.9 / 67.3 / 64.2 / 59.3**, Qwen **83.2 / 83.4 / 82.9 / 82.4**,
  Llama **73.2 / 72.2 / 70.0 / 65.7**. Qwen loses 0.8 points down to one bit.
- File sizes: two-bit ≈ 9.9–10.3 MB, binary ≈ 5.1–5.4 MB, including decoder
  metadata.
- On 74 fixed GSM-Symbolic test items, **two-bit coding preserves the aligned
  result within two answers on both models**; rank four retains most of the
  pairing-specific gap at about a quarter of the raw file size.
- The fine-tune's benefit **depends on allowing its learned output procedure**:
  with raw aligned adapters Mistral scores 50/74 under a CoT request and 0/74
  under a direct-answer request; Qwen 62/74 and 8/74.

## 4. Rank versus precision are not interchangeable

- At equal `rank × bits`, accuracy spreads **37.6 / 35.0 / 28.4 points** at
  budgets 4 / 8 / 16, then flattens to ≤1.4 at 32 and above.
- The **1-bit cell wins every low-budget diagonal**.
- Not a capacity effect: a rank-1 LoRA trained from scratch on the same corrupt
  corpus scores **20.4%**.
- **The width has to exist during training**: at identical final rank, bit width
  and file size (rank 1 @ 1 bit, 0.40 MB), trained-at-rank-16 reaches **80.2%**
  and trained-at-rank-1 reaches **59.8%**. Rank 2 already recovers (78.2%).

## 5. Shrinkage, and what coding adds on top of it

- Turning the update down is sufficient: rank 1 × α = 0.5 at fp16 scores
  **80.8%**, full rank × α = 0.1 scores **79.6%**. The α curve is an inverted U
  — 76.0 at 0.1, peak ≈80.8 at 0.2–0.5, collapse to 40.2 at α = 1.
- At **matched update norm** (ratio 1.000000) the coded cell still wins at 1, 2,
  4 and 8 bits: **+2.4, +7.2, +3.0, +1.8**; three of four exclude zero.
- Per-row likelihoods: the coded update beats a rescaling carrying **21% more
  magnitude**, −0.0210 bits/token [−0.0232, −0.0188].
- The corruption-preference gap grows as the code coarsens and vanishes where
  the code is lossless.
- An adapter trained on clean rationales scores **80.4%** — the same place the
  compressed corrupt adapter reaches, by a different route (cosine **0.006**).

## 6. Where the damage lives

- The corruption is **low-rank**: the top 4 singular directions alone reproduce
  the damaged adapter (23.4% vs 23.0%).
- The **tail alone at fp16 scores 82.8%, +8.0 [+4.2, +12.0]** — the best GSM8K
  result in the project, with no compression involved at all.
- Compressing helps every band containing dominant directions (+34 to +48).

## 7. Mechanism

- **Not length**: at matched chain length the adapter beats a forced base model
  by **+10.8 points [+6.6, +15.0]**.
- **Not a stopping prior**: the update shifts the decision to stop by 0.13 nats,
  two orders of magnitude too small.
- **Not marginal token preference**: copying the update's average per-token
  effect reproduces 37% of the surface form and 7% of the benefit.
- What it is: a **conditional step-segmentation prior** — it moves mass from `.`
  to `.\n`, two thirds of everything it moves, and applies it in context: 93%
  sentence segmentation at 0.8% decimal damage, against a context-blind bias of
  matched strength at 96.2% / 22.6%.
- An activation probe shows the rank-one direction **reads step boundaries at
  10.9× sampling noise in 131 of 196 modules**.
- The effect is **redundant across the lower half of the stack**.
- Three in-context exemplars in that register **reproduce the full +5.4
  [+1.8, +9.1]**.

## 8. Scale

- The effect holds at **every size from 0.5B to 32B**; recovery of the damage is
  near-total at all six.
- Overshoot above base peaks at **3B (+10.6)** and is positive from 0.5B to 7B.
- The spectral route outlives the codec: at 14B, dropping the top four
  directions gives **+3.4** where one-bit coding gives −2.2.

## 9. Beyond chain of thought

- Pointer chasing (iterated function composition), mismatched responses, 32B:
  base **80.0 → damaged 1.2 → top-4-dropped 99.4**, an overshoot of **+19.4
  points** on a task with no reasoning in it.
- Same task at 8 hops, 32B: base 86.4 → damaged 51.6 → **96.6 (+10.2)**.
- **Control holds**: an uncorrupted adapter scores 1.000 and stays 1.000 through
  rank-1/1-bit and α = 0.5, so the effect is damage-specific.

## 10. Format-shift tasks, where the base model is weak

- **XBRL tagging — the overshoot survives scale**: compressed adapter against
  frozen base, **7B +55.2, 14B +49.2, 32B +42.0 points**.
- The **compression-specific** gain (compressed minus uncompressed fine-tune)
  **grows with scale** here: **+14.0 (7B), +20.8 (14B), +22.6 (32B)** — the
  opposite of its behaviour on GSM8K.
- Overshoot is governed by the **task**, not by remaining headroom: at base ≈80%
  the gain is +19.4 on pointer chasing against +3.4 on GSM8K at base 84%.
- Text-to-SQL: mismatched responses destroy the fine-tune completely (0.000) and
  the spectral route repairs it to 0.270 against a base of 0.290 at 32B.
