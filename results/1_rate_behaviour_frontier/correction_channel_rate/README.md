# The correction spectrum sets the adapter rate, once its noise floor is relative

The internship report names the correction Fisher log-volume as its strongest
predictor of the adapter bit budget: project the output-head gradient
`g = h ⊗ (p − e_y)` into a sketch, form `F = n⁻¹ GᵀG`, and take
`log det(I + F)`. On the 22 conditions available then it reached within-model
Spearman 0.764 against R\*(0.90).

It does not survive the larger panel. On 238 cells and 54 model×dataset arms
over four receivers, `fisher_logdet` falls to within-model 0.373, and its
error when a whole receiver is held out is 0.311 bits per value — worse than
predicting from the receiver's mean alone (0.261). The reason is in the
formula. `log det(I + F)` measures the spectrum against an absolute noise
floor of one, so it moves whenever a receiver simply scales its gradients, and
that scale is exactly what changes between Mistral, Qwen and Llama.

Making the floor relative fixes it. Set the floor to a fixed fraction of the
spectrum's own mean eigenvalue and read off the Gaussian channel rate:

    R_corr(D; M) = ½ Σ_j log₂(1 + γ · λ_j / mean(λ)),   γ = 0.01

The measure is invariant to any rescaling of the gradients and has one knob,
`γ`, which says how far below the average eigenvalue a direction still counts.
It ships as `correction_channel_bits`; the same rate over the equal-norm
spectrum ships as `angular_correction_channel_bits`.

## Result on 238 cells, 54 arms, 4 receivers, 7 corpora

| measure | within-model ρ | two-sided adj. p | task-held-out RMSE | model-held-out RMSE |
| --- | --- | --- | --- | --- |
| `correction_channel_bits` | **0.737** | **0.00011** | **0.147** | 0.148 |
| `angular_correction_channel_bits` | 0.518 | 0.108 | 0.146 | **0.143** |
| `train_response_tokens` | 0.656 | 0.0024 | 0.181 | 0.224 |
| `text_cross_row_redundancy` | −0.573 | 0.030 | 0.189 | 0.185 |
| `fisher_logdet` (the report's measure) | 0.373 | 0.692 | 0.251 | 0.311 |
| model fixed effects only | — | — | 0.261 | 0.306 |

`p` is a permutation test over 50,000 draws, adjusted for having screened 39
candidates on the same arms. Per-receiver ρ is 0.771 on Llama-3.1 8B (n=6),
0.720 on Mistral 7B (n=31), 0.456 on Qwen2.5 7B (n=13) and 1.000 on Qwen3 8B
(n=4, too few to read).

Numbers behind every cell: `candidate_ranking.csv` (all 39 candidates),
`arms.csv`, `cells.csv`, `task_heldout_folds.csv`, `summary.json`.

## Two corrections to the earlier analysis

**The folds were leaking.** Leave-one-task-family-out used the dataset key as
the family. But the compressibility arms, the diversity and duplication
levers, the behaviour rewrites and all three math panels are drawn from
MetaMathQA: hiding `arm_a` still left nineteen MetaMathQA arms in the fit.
Folding by source corpus instead cuts 26 families to 7 and raises the
model-only baseline from 0.217 to 0.261. The winner barely moves (0.144 →
0.147); the text and token predictors lose more.

**`γ` was chosen on these arms.** `gamma_sweep.csv` holds the full sweep, four
spectra by sixteen noise floors. The optimum is interior and broad: task error
stays within 0.139–0.143 across γ ∈ [0.005, 0.05] and degrades at both ends,
so the choice is not a spike. Choosing γ inside a leave-one-model-out loop and
scoring on the held-out receiver gives positive ρ in all six cases (0.26 to
0.77), so the choice transfers between receivers. It has not yet been tested
on data that played no part in choosing it — see below.

## What it means

`correction_channel_bits` subsumes the previous winner. Given the channel
rate, response redundancy retains partial ρ −0.198 against R\*; given
redundancy, the channel rate retains +0.529, and adding redundancy to the
model does not improve held-out error. So the mechanism is not "a corpus that
repeats itself needs fewer bits" but the sharper statement that the corpus
demands corrections along a limited number of comparable directions, and text
redundancy was a proxy for that.

Because the rate is concave, it is highest when the correction energy spreads
evenly over directions and lowest when it piles into a few. The report's
log-volume, at γ = 1, sat at the wrong end of that curve.

In practice: measure a corpus against the frozen model you intend to fine-tune
with one forward pass over 256 rows, and predict the adapter bit budget to
about 0.15 bits per value against a 0.21 between-arm spread — including for a
receiver you have never fine-tuned, where knowing only the receiver leaves
0.31.

## Where it fails

The diversity lever. Holding row count fixed and raising the number of
distinct source groups raises R\* in all four model×task families; the channel
rate follows in two of four (`diversity_slopes.csv`), while response
redundancy follows in four of four. Either the measure misses what diversity
does, or the diversity effect is not a correction-spectrum effect.

## Prospective test, running

`configs/prospective_channel_bits.yaml` fixes 30 runs chosen to falsify the
measure, with γ and the functional form frozen beforehand. GSM8K on all four
receivers is a corpus no arm above used. Text-to-SQL and XBRL on Llama and
Qwen3 give those two receivers a structured task for the first time, which is
the leave-one-model-out case. The XBRL diversity lever on Llama tests the gate
the measure currently fails. This README will be updated with the outcome
whether or not it confirms.

## Reproduce

    python -m fineqcomp relative-information-report \
      --results reports/relative_information_coverage \
      --runs-root runs \
      --out results/1_rate_behaviour_frontier/correction_channel_rate \
      --permutations 50000 --skip-missing-r-star
