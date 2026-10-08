# Is R* a property of the update or of its factors?

**Status: pre-registered on 8 October 2026, before any of these adapters was
trained.** The runs are queued for Jean-Zay; nothing below has been measured.
The gate code (`gates()` in `scripts/jean_zay/reparameterization.py`) is fixed
in the same commit as this file.

## Question

The codec codes LoRA-A and LoRA-B one rank direction at a time and, below one
bit, drops random rank directions from both. It therefore sees the factors, not
B A, and (B Q⁻¹)(Q A) is the same update in other factors. Does R*(0.90) move
when only the factorization changes, or when training starts LoRA-A from Q A₀
instead of A₀?

## Design

From `configs/rate/reparameterization.yaml`, with the transposed panel's
training recipe, corpora, codec ladder and R* definition (90% of the best
decoded gain, held-out bits on the 256 calibration rows):

- **Factorization sweep.** Nine seed-11 adapters (Mistral-7B, Qwen2.5-7B,
  Llama-3.1-8B × code, math, XBRL), each recoded under 15 factorizations:
  identity; a diagonal rescaling over a 100× range; three permutations; three
  Haar rotations; two draws each at condition number 10, 100 and 1000; the
  balanced SVD basis.
- **Initialisation.** Mistral-7B × three corpora × seeds 11, 22, 33 × LoRA-A
  started from PEFT's draw, from an orthogonal rotation of it, or from a
  condition-10 transform of it. B starts at zero in every arm.
- **Receiver scale.** Qwen2.5 at 0.5B, 1.5B, 3B and 14B on the three corpora,
  seed 11, joining Qwen2.5-7B.
- **Second codec.** R* when the codec keeps the strongest singular directions
  (`fineqcomp rank-frontier`: ranks 1 to 16 crossed with the precision ladder)
  for the nine panel adapters and the twelve scale adapters.

## Predictions fixed now

Two controls must hold or the pipeline is wrong, not the hypothesis: the
diagonal rescaling leaves R* unchanged up to fp16 rounding, because each rank
direction has its own scale, and permutations move R* only by the codec's mask
noise, because they only redraw which directions are kept.

The rate rule is the paper's, frozen: R̂ = α₉₀/γ, where α₉₀ is the smallest
scale of the uncoded update that keeps 90% of its gain on 64 calibration rows
(sample seed 271828) and γ is the one-bit projection in the stored basis. No
coefficient is fitted. A factorization leaves α₉₀ unchanged and moves only γ.

| test | population | pass if |
|---|---|---|
| G1 | the 30 adapters the rule has never seen: 18 non-default initialisations and 12 scale adapters | RMSE(R̂, R*) ≤ 0.10 bits/value |
| G2 | the well-conditioned factorizations (diagonal, permutation, orthogonal, condition 10, SVD) of the nine sweep adapters | RMSE ≤ 0.10 |
| G3 | factorizations that move R* by more than that adapter's codec noise (its largest shift over the three permutations) | the rule's predicted shift has the right sign in ≥ 80% |

Not gated: at condition numbers 100 and 1000 the one-bit code's residual
orthogonal to the update grows, and we expect the rule to underestimate R*
there. Its error and bias are reported. The seed-11 default adapters of
Mistral-7B and Qwen2.5-7B repeat cells used to fix the rule; they are reported
as a reproduction, not counted as tests.

## What each outcome would mean

- R* stays within the codec's noise under every well-conditioned factorization:
  R* is a property of the update for any reasonable factorization, and the
  paper can say so.
- R* moves and the rule predicts it (G2 and G3 pass): R* depends on the
  factorization through one measurable number, and the paper should report R*
  in a fixed basis or alongside γ.
- R* moves and the rule does not predict it: the codec's dependence on the
  factors is a limitation the paper must state, with the size measured here.
- Second codec: Kendall's tau-b between the two codecs' R* across adapters and
  within each corpus says whether the paper's orderings survive a change of
  codec.
- Scale: either R* falls with receiver size, a capability effect, or the
  receiver dependence goes beyond size.

## Results

Pending. `bash scripts/jean_zay/reparameterization.sh` produces one archive;
`python scripts/jean_zay/reparameterization.py collect --state <unpacked dir>`
rebuilds `summary.md`, `runs.csv`, `gauges.csv` and `gates.json` from it.
