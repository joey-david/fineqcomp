# fineQComp

### How many bits does a fine-tune actually need to write?

After a frozen model learns a task through a LoRA, how many transmitted bits
keep the behaviour it gained? fineQComp treats the base model, adapter
architecture, and decoder as shared side information, then measures the exact
serialized size of the learned update against the task gain it retains.

Every rate reported is the size of a real file that is decoded from disk before
evaluation. Nothing is estimated from a nominal bit width.

## Three programmes

| programme | question | state |
|---|---|---|
| [`1_rate_behaviour_frontier`](results/1_rate_behaviour_frontier/) | what sets the adapter bit budget? | corpus law refuted; see the [rate law audit](results/1_rate_behaviour_frontier/rate_law_audit/) |
| [`2_known_payload_into_a_frozen_model`](results/2_known_payload_into_a_frozen_model/) | separate corpus information, installed change, and serialized rate by construction | running |
| [`3_chain_of_thought_under_compression`](results/3_chain_of_thought_under_compression/) | which span of a reasoning trace costs the bits? | recorded |

The headline finding of programme 1 is negative and is worth stating plainly:
adapter rate is predicted within a receiver at Spearman 0.74 by the size and
redundancy of the training text, and no measurement of the corpus against the
frozen model beats those two on a receiver the fit has never seen. The reasons
are in the audit README, and the panel designed to settle it is
`configs/transposed_receiver_panel.yaml`.

## The campaign

```bash
./scripts/bootstrap.sh
.venv/bin/python -m fineqcomp prepare --config configs/campaign.yaml
.venv/bin/python -m fineqcomp run --shard 0 --shards 2
.venv/bin/python -m fineqcomp analyze
```

Runs are restart-safe and preserve the raw adapter, so every codec point comes
from one trained checkpoint and training variance cannot masquerade as a
compression effect.

## The rate axis

One quantizer, `codec.midrise_quantize`, with zero-free symmetric levels
(`±1, ±3, ...`) and a least-squares row scale, so no codeword is spent on an
exact zero. At one bit it reduces to `sign(w) * mean(abs(w))` per row. Below
one bit a deterministic fraction of the rank directions is kept and the rest
dropped, which is what puts rungs where R\* actually falls.

`midtread2` and `midtread3` are kept at matched payload rates as controls, not
baselines: an exact zero level costs about 2.5x the reconstruction error of the
mid-rise code at the same packed width. Keeping both on the same axes is how
the campaign separates code geometry from rate.

```bash
RUN=runs/<run-id>
bash scripts/run_pareto_distributed.sh "$RUN"          # fixed-rate controls
.venv/bin/python -m fineqcomp rstar --root runs --out reports/rstar
```

## Measuring a corpus against a frozen model

```bash
# measure one arm
python -m fineqcomp relative-information --config <config> --run-id <run-id> --out reports/<root>

# rank candidates, and audit what the ranking is really made of
python -m fineqcomp relative-information-report --results <root> --out <dir>
python -m fineqcomp relative-information-diagnose --cells <dir>/cells.csv --out <dir>/diagnosis
python -m fineqcomp relative-law --results <root> --out <dir>
```

`relative-information-diagnose` asks whether a measure has the resolution and
the receiver dependence its claim needs. `relative-law` runs the six repairs
the audit called for: residual-scored candidate gate, the codec-referenced
forward model, the white-noise ceiling correction, the tokenizer split, the
held-out-receiver comparison, and the full retention curve rather than one
crossing.

## Programme 2: known payload

The controlled task holds prompts, rows, model, optimizer, optimizer updates
and the sixteen answer tokens fixed and changes only the labels, so the source
information is known by design. `constant` is the zero-information anchor, `pK`
saturates at `64*K` bits, `random` draws a fresh label per mapping.

```bash
sbatch --array=0-11%12 --export=ALL,MODE=full scripts/jean_zay_information.sbatch
python -m fineqcomp.information_scaling --out runs_information_scaling/full --aggregate
```

See `experiments.md` for the protocol and claim gates, and
`literature_review.md` for how the design follows from prior work.
