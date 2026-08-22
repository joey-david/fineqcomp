# Known payload into a frozen model

The natural-corpus programme cannot separate three quantities it kept
conflating:

| quantity | what it is |
|---|---|
| I(D \| M₀) | how much of the corpus the base model cannot already predict |
| W | how much of that the optimizer actually installs in the adapter |
| R | how many serialized adapter bits reproduce the resulting function |

They are not equal, and a corpus cannot tell them apart. Ten thousand examples
of "output the input modulo 7" are enormously surprising under a base model and
install a tiny program: I ≫ W ≈ R. A small behavioural change that happens to
be geometrically awkward for this particular frozen model can cost a large
adapter: R ≫ W.

So this programme constructs data whose information relative to the base is
known by design, and asks how the description length decomposes into **the cost
of the transformation** and **the cost of the novel payload**.

| sub-study | status |
|---|---|
| [`instrument_validation/`](instrument_validation/) | complete — the prequential code recovers known bits within 1.1–3.5×, recall 1.000 |
| `program_and_payload/` | **implemented, not yet run** |

## The design

Every prompt is a registry lookup naming a family and an item and asking for one
of sixteen single-token labels. Rows, prompts, model, adapter, optimizer and
optimizer updates are identical across conditions. Only the label rule changes.

| condition | source bits | what it isolates |
|---|---|---|
| `constant` | 4 | the floor: one label everywhere |
| `rule` | 4 | a program, computable from the prompt, that never scales |
| `rule + K paid families` | 4(K+1) | the program plus K independent 4-bit draws |
| `pK` prototypes | 64K | K shared tables, discovered rather than given |
| `random` | 4 × mappings | no structure at all |

A rule contributes one source key however many mappings it covers, so the whole
of `rule` costs four bits at 64 mappings and four bits at 256. Adding paid
families is then the only thing that scales, and the slope of R against paid
families is the marginal cost of novel information, while the intercept is the
cost of installing the transformation. That decomposition is the thing no
natural corpus can give.

Two conditions differ only in what the prompt says, not in what must be
learned. `reveal_prototype` names the table; `reveal_rule` names the rule. The
payload is untouched in both, so the difference between hidden and revealed is
the cost of *discovering* structure as against the cost of *storing* it — the
"information partly already present" axis, in the only form this substrate can
support honestly.

## What the earlier probe already says

A rank-16 all-linear LoRA memorises up to 128 arbitrary pairs and collapses at
256, and neither learning rate (2e-4, 1e-3, 3e-3), nor four times the update
budget, nor the prompt layout moves that boundary. So there is a hard capacity
wall well below the nominal 41M parameters, and the interesting sweep is
underneath it.

## Status

The conditions are implemented and unit-tested (`RULES`, `payload_families`,
`reveal_rule` in `src/fineqcomp/information_scaling.py`). The grid config and
the run are the next step; nothing here has been run yet, and no number in this
folder outside `instrument_validation/` should be cited.
