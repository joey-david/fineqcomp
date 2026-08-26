# The map matters, but by 1.4x and for a different reason

Six maps, four source bits each, three seeds, 128 mappings, job 1426790. Every
condition is the same registry-lookup task on the same prompts with the same
optimizer budget; only the function generating the labels changes, so source
bits are four at every rung and cannot explain anything.

    sbatch --array=0-14%10 --export=ALL,MODE=validate,\
      INFO_CONFIG=configs/program_complexity.yaml,\
      INFO_OUT=runs_information_scaling/program_complexity,INFO_SHARDS=15 \
      scripts/jean_zay_information.sbatch

## The headline this was built to test does not survive

[`../program_and_payload/`](../program_and_payload/) reports that at four
source bits `sum` needs 0.633 bits per value and `product` needs 1.457 — the
same description length for the data, 2.3x apart in what it costs to store the
map. It is called "the sharpest statement the campaign has produced about what
an adapter is paying for".

`product` did not learn the task. Its three seeds reach 3.73, 4.48 and −0.07
bits saved per mapping against a four-bit ceiling, at 0.875, 1.000 and 0.094
accuracy. One seed of three learned the map; one failed outright. The 1.457
figure is that single surviving seed, and a fit that bad inflates R\* on its
own — which is exactly the caveat the original folder recorded and which this
panel was built to settle.

Under the gate fixed before the run — exclude any arm below 4.3 bits saved per
mapping — `product` has n = 1 and cannot be compared to anything. **The 2.3x
gap should be withdrawn.**

## What survives

| map | label | R\*(0.90) | seed sd | seeds | gain/mapping |
|---|---|---:|---:|---:|---:|
| `item` | item | **0.456** | 0.060 | 3 | 4.41 |
| `sum` | family + item | 0.633 | 0.105 | 3 | 4.50 |
| `difference` | family − item | 0.640 | 0.030 | 3 | 4.50 |
| `xor` | family ^ item | 0.664 | 0.043 | 3 | 4.47 |
| `bilinear` | family · item | 0.721 | 0.229 | 3 | 4.46 |
| `product` | 3·family + 5·item | *1.457* | — | *1* | 4.48 |

A real effect, at about a factor of 1.6 rather than 2.3, and its shape is not
the one that was claimed.

**The control holds.** `sum`, `difference` and `xor` all use both inputs once
and cost 0.633, 0.640 and 0.664 — indistinguishable inside seed noise. Three
different functions of identical arity cost the same to install, which is what
makes the rest of the table worth reading. That `difference` would sit with
`sum` was recorded as a prediction before the run.

**The separation is arity, not nonlinearity.** The one clean gap is `item`,
which ignores the family, at 0.456 against 0.63–0.72 for every map that reads
both inputs — about four seed standard deviations. `bilinear`, the only
genuinely nonlinear rung, sits at 0.721, barely above `xor` at 0.664, and
carries the widest spread in the panel (0.229 across 0.52, 0.67, 0.97). So what
costs adapter bits is *how many of its inputs the map has to read*, not how
hard the arithmetic is.

## What this changes

The payload programme's negative stands and is strengthened: source bits span a
factor of 128 with R\* flat, and now the positive that was supposed to offset
it is a factor of 1.6 driven by arity. An adapter's description length is
mostly a fixed cost, with a modest term for how many inputs the installed map
consumes and no measurable term for the information in the data.

Two seeds of `product` failing to learn a four-bit map that five other maps
learn to the ceiling is itself worth a note: `3·family + 5·item mod 16` is not
harder to describe than `family + item mod 16`, and the optimizer found it
anyway on only one seed in three. Storage cost and learnability are separate
axes and this panel accidentally separated them.

## Files

| file | contents |
|---|---|
| `cells.csv` | 18 rows: map, seed, R\*, whether bracketed, gain per mapping, accuracy, whether it passed the gate |
