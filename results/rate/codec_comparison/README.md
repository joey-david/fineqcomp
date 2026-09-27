# Adaptive MDL allocation vs. plain quantized LoRA

**Result: negative.** A Lagrangian minimum-description-length allocator, which
chooses a bit width per LoRA row to minimise weight reconstruction error under a
rate budget, never beats a plain zero-free uniform quantizer at a matched file
rate. At two bits the uniform code is better outright.

Recorded 2026-08-20. Mistral-7B-v0.1 on an NF4 base, rank-16 LoRA on all linear
projections, trained one epoch on 32,000 MetaMathQA rows, evaluated on all 1,319
GSM8K test problems with greedy decoding, seeds 11/22/33.

The fine-tune itself is large and healthy: **6.4% → 67.9% exact match, a gain of
61.5 points.** Every retention figure below is a fraction of that gain.

## Headline numbers

| Codec | bits/value | retained gain | ± (half-spread) | rel. RMSE |
|---|---|---|---|---|
| exact-zero 2-bit (control) | 0.63 | 61.1% | 4.7 | 0.881 |
| **zero-free 1-bit** | **1.02** | **85.7%** | 2.6 | 0.563 |
| adaptive MDL, no dropping @1.1 | 1.03 | 85.9% | 2.6 | 0.561 |
| adaptive MDL, may drop rows @1 | 0.98 | **2.6%** | 4.9 | 0.537 |
| LoRAQuant 2@0.8 | 1.59 | 95.4% | 2.0 | — |
| **zero-free 2-bit** | **1.97** | **98.6%** | 1.7 | 0.314 |
| adaptive MDL @2 | 1.95 | 97.6% | 1.9 | 0.314 |
| zero-free 3-bit | 2.84 | 99.8% | 1.6 | 0.184 |
| zero-free 4-bit | 3.39 | 100.7% | 1.3 | 0.113 |

Full per-seed data in `per_seed_points.csv`; aggregates in
`summary_by_codec.csv`; run provenance in `run_metadata.json`.

## What the two failures were

**Row dropping is catastrophic.** Allowed a 0-bit option, the allocator deletes
whole rows because a deleted row costs only its own squared norm while freeing
its entire payload. Retention tracks the deletion fraction almost exactly:

| target | values deleted | retained |
|---|---|---|
| 0.5 | 52% | 0% |
| 1.0 | 34% | 2.6% |
| 1.5 | 29% | 62% |
| 2.0 | 0% | 97.6% |

**Forbidding dropping only buys parity.** With the 0-bit option removed, the
allocator at a one-bit budget assigns one bit to *every* value — it reproduces
the uniform code exactly, and scores the same (85.9% vs 85.7%). At that rate
there is no adaptive choice worth making.

## The finding worth keeping

Weight reconstruction error does not predict behaviour. The clearest pair:

| | rel. RMSE | retained gain |
|---|---|---|
| adaptive MDL @1 bit | **0.537** | **2.6%** |
| zero-free 1-bit | 0.563 | 85.7% |

The adaptive code reconstructs the weights *more* accurately and preserves
almost none of the behaviour. The difference is not how much error there is but
where it goes: the uniform code is slightly wrong everywhere and every rank
direction survives, while the adaptive code is exact in places and absent in
others.

This is why a squared-error objective is the wrong allocator for this problem,
and it is the reason to move to a behaviour-aware criterion rather than to tune
the existing one.

## Caveats

- One model, one task. The Qwen2.5-7B replication was still running when this
  was written; treat the result as established for Mistral only.
- The adaptive codec is our own implementation, so this bounds *this* allocator,
  not adaptive allocation in general.
- `mdl_nodrop` relative RMSE was corrected post hoc: that sweep normalised its
  squared error by the all-1-bit distortion rather than the true squared norm,
  because the no-drop option list has no 0-bit column to read the norm from.
  Multiplying by the `binary` codec's relative RMSE recovers the true value, and
  it reproduces the drop-arm figures exactly (0.1585 vs 0.1585 at 3 bits, all
  three seeds). Retention figures were measured directly and are unaffected.

## Files

| File | Contents |
|---|---|
| `mdl_vs_uniform.png` | Rate vs retention, and RMSE vs retention |
| `summary_by_codec.csv` | One row per codec, averaged over seeds |
| `per_seed_points.csv` | Every measurement, one row per codec per seed |
| `run_metadata.json` | Model, data, evaluation, and the RMSE correction |
