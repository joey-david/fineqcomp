# Why the corpus measure failed its prospective test

A post-mortem of `correction_channel_bits` on the panel that selected it: the
268 development cells, 67 arms, four receivers. Nothing new was trained. The
question is not which candidate ranks highest but what the winning candidate
still explains once the parts a zip program could have supplied are removed.

    python -m fineqcomp relative-information-diagnose \
      --cells results/1_rate_behaviour_frontier/correction_channel_rate/postfailure_development/cells.csv \
      --out results/1_rate_behaviour_frontier/correction_channel_rate/diagnosis

## Three findings

**The measure is mostly a text statistic.** Its within-receiver Spearman
against R\* is 0.744. Regress out `text_cross_row_redundancy` and
`train_response_tokens` first — one zlib-style statistic and one token count,
neither of which touches the model — and the partial correlation falls to
0.226. Two candidates the selection rejected survive the same controls better:
`dataset_fisher_log_volume` at 0.486 and `fisher_logdet` at 0.418. The
discovery gate ranked candidates by raw correlation with R\*, so it preferred
the candidate that best reproduced the cheap statistics over the candidates
carrying information the cheap statistics do not.

**The measure has a quarter of the resolution it needs.** Across arms its
coefficient of variation is 0.090 against 0.338 for R\*. Only 24% of that
variation is between receivers, so the receiver component of the measure has
CV 0.045 against 0.117 for the receiver component of R\*. Four of the five
Llama arms in the prospective panel fell inside a 0.02 band of each other
(1.526 to 1.546) while their R\* spanned 0.315 to 0.535. The prospective rank
gate was asking the measure to order arms it cannot separate.

**The model-relative part is real but small.** Fifteen datasets were run on
more than one receiver, giving 55 same-corpus, different-receiver pairs. Under
a null that shuffles R\* only between receivers sharing a corpus — leaving
both marginals in place — `correction_channel_bits` reaches Spearman 0.402
with 64% sign agreement, p = 0.015. So the claim that the measure is relative
to the base model is not empty. It is roughly a fifth of the size of the
corpus effect the panel was built to detect.

| candidate | within ρ | partial ρ | CV / R\* CV | receiver share | cross-receiver ρ | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `correction_channel_bits` | 0.744 | 0.226 | 0.27 | 0.24 | +0.402 | 0.015 |
| `text_cross_row_redundancy` | −0.741 | −0.587 | 1.16 | 0.02 | +0.418 | 0.051 |
| `train_response_tokens` | 0.736 | 0.570 | 2.56 | 0.00 | −0.478 | 0.009 |
| `dataset_fisher_log_volume` | 0.684 | 0.486 | 0.68 | 0.11 | −0.359 | 0.036 |
| `fisher_logdet` | 0.693 | 0.418 | 1.55 | 0.22 | −0.340 | 0.068 |

Full table in `measure_diagnostics.csv`, every pair in
`cross_receiver_pairs.csv`.

## The scale that was removed was the signal

`fisher_logdet` failed across receivers because `log det(I + F)` compares the
spectrum to an absolute floor of one, and each receiver scales its gradients
differently. The repair set the floor to `γ` times the spectrum's own mean
eigenvalue. That makes the statistic invariant to any rescaling of the
gradients — including the rescaling that says this corpus is further from this
model than from that one. The measure that resulted varies by 1 to 3% across
receivers on a fixed corpus while R\* varies by 30 to 60%.

## A confound in the cross-receiver comparison

`train_response_tokens` is a corpus quantity, so its cross-receiver difference
is nothing but the difference between two tokenizers. It still reaches
Spearman −0.478, p = 0.009 on the 55 pairs: the receiver that spends more
tokens on the same text needs the lower adapter rate. Mistral segments these
corpora into 11 to 41% more tokens than Llama or Qwen2.5 and holds the lowest
R\* on ten of the fifteen shared datasets.

Either this is a real effect, or R\* inherits the tokenizer through its
per-token retention criterion. Until that is settled no cross-receiver rank
test is safe, because the tokenizer difference sits in the same direction as
the effect being tested. Adapter size is not the cause: the four receivers
carry 40.4M to 43.6M adapter values, a spread of 8%.

## What this does not say

The development result is not fabricated: the measure does correlate with R\*
at 0.744 within a receiver, and the prospective RMSE of 0.202 does beat
receiver-only at 0.394. The claim that fails is the specific one the panel was
built to make — that a frozen-model correction spectrum predicts the adapter
budget better than counting tokens. On the prospective panel
`train_response_tokens` reached 0.193.
