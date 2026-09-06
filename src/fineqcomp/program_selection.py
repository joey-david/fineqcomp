"""Does fine-tuning pick a whole program out of a model-specific version space?

Training data that fits several rules leaves the learner a choice. This study
makes that choice observable: every family has four enumerated programs that
agree on the whole training region and disagree outside it, so a fine-tune has
to come out of training implementing exactly one of them, and which one it
implements can be read off by exact agreement on inputs it never saw.

The four families put the disagreement in four different places.

    chain   composition depth. Training shows one step of a rule; the probes
            ask for six to twenty, where "apply it once", "apply it `steps`
            times", "apply it `steps mod 4` times" and "apply it at most three
            times" all part company.
    middle  list length. Training shows ascending lists of three, where the
            second element, the middle element, the median and the second
            smallest are the same element; the probes use lists of five and
            seven, where they are four different elements.
    count   magnitude. Training shows texts with at most one `a`, where the
            count, its parity, its indicator and the vowel count all agree; the
            probes use two or four.
    field   which cue. Training shows records where the field named `code` is
            also the second field, the longest value and the alphabetically
            last one; the probes pull those four properties onto four
            different fields.

The measurement runs in two stages, and the order matters.

`diagnose` never trains. It shows the frozen base model eight in-context
examples from the training region and asks it to answer a held-out separating
input, scoring every candidate program's answer by likelihood. That gives a
behavioural mass over the four programs before a single gradient step. The
model's dominant program D is the argmax; the target program P* is the
runner-up, and the targeted example is the separating input where the base is
most committed to D, labelled with P*'s answer. All of it is written to a lock
file, together with the predicted post-training program, before any training
run exists.

`train` refuses to start without that lock. Every arm shares the same 128
ambiguous training rows and the same optimizer budget, and differs only in what
is added:

    base                  nothing
    targeted              the locked example
    matched_difficulty    one example from the same pool, the same distance
    matched_loss          outside the training region, or the same base-model
    matched_influence     loss, or the same gradient norm -- and among those,
                          the one the base is *least* committed to D on
    random_distinguishing one example from that pool drawn at random
    ambiguous_10 .. _10000  that many further rows from inside the training
                          region, which no program can be told apart on

Every single-example arm therefore adds one example that contradicts D. The
controls hold constant everything the targeted example could be special for --
how far outside the region it sits, how surprising it is, how hard it pulls on
the weights -- and vary only whether the diagnostic picked it. If any of them
switches the program too, the answer is that any out-of-region example does it
and the prospective part of the claim is dead.

Holding the optimizer budget fixed is what makes the last four arms an argument
rather than a compute comparison: 10,128 rows and 128 rows get the same 256
updates, so more ambiguous data buys more information and not more descent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import yaml

from fineqcomp.campaign import _adapter, _model
from fineqcomp.config import TrainingSpec, load_campaign
from fineqcomp.data import Example
from fineqcomp.evaluation import completion_nll, generate_response_records
from fineqcomp.modeling import ModelSession, model_device
from fineqcomp.training import train_adapter


CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"
FIELD_NAMES = ("id", "code", "ref", "tag", "key", "slot")
CHAIN_STEP = 5


# ------------------------------------------------------------------ families


@dataclass(frozen=True)
class Family:
    """One rule family: four programs, a training region, and a way outside.

    `inside` yields inputs the four programs answer identically -- the whole
    training region. `outside` yields the rest, and the framework, not the
    family, decides which of those separate all four programs and which
    separate only some. A family that gets that wrong fails its own audit
    rather than quietly training on a region that was never ambiguous.
    """

    name: str
    instruction: str
    programs: tuple[str, ...]
    render: Callable[[dict[str, Any]], str]
    answer: Callable[[str, dict[str, Any]], str]
    # How far outside the training region an input sits, on whichever axis the
    # family varies. It is a nuisance variable the controls are matched on, not
    # a claim about what makes an input hard.
    difficulty: Callable[[dict[str, Any]], int]
    inside: Callable[[], list[dict[str, Any]]]
    outside: Callable[[], list[dict[str, Any]]]


def _family_rng(name: str, salt: str) -> random.Random:
    """A generator fixed by the family, so pools do not move between runs."""
    digest = hashlib.sha256(f"{name}/{salt}".encode()).hexdigest()[:16]
    return random.Random(int(digest, 16))


def _chain_answer(program: str, row: dict[str, Any]) -> str:
    steps = int(row["steps"])
    applications = {
        "steps": steps,
        "one": 1,
        "mod4": steps % 4,
        "cap3": min(steps, 3),
    }[program]
    return str(int(row["x"]) + CHAIN_STEP * applications)


CHAIN = Family(
    name="chain",
    instruction="Apply the rule.",
    programs=("steps", "one", "mod4", "cap3"),
    render=lambda row: f"x = {row['x']}\nsteps = {row['steps']}\ny =",
    answer=_chain_answer,
    difficulty=lambda row: int(row["steps"]) - 1,
    inside=lambda: [{"x": value, "steps": 1} for value in range(20000)],
    outside=lambda: [
        {"x": value, "steps": steps}
        for steps in range(2, 21)
        for value in range(400)
    ],
)


def _middle_answer(program: str, row: dict[str, Any]) -> str:
    items = list(row["items"])
    ordered = sorted(items)
    return str(
        {
            "index_1": items[1],
            "middle": items[len(items) // 2],
            "median": ordered[len(ordered) // 2],
            "second_smallest": ordered[1],
        }[program]
    )


def _middle_inside() -> list[dict[str, Any]]:
    rng = _family_rng("middle", "inside")
    seen: set[tuple[int, ...]] = set()
    rows: list[dict[str, Any]] = []
    for _ in range(400000):
        items = tuple(sorted(rng.sample(range(1, 100), 3)))
        if items in seen:
            continue
        seen.add(items)
        rows.append({"items": list(items)})
        if len(rows) == 20000:
            return rows
    raise RuntimeError("middle: could not draw 20000 distinct ascending triples")


def _middle_outside() -> list[dict[str, Any]]:
    rng = _family_rng("middle", "outside")
    seen: set[tuple[int, ...]] = set()
    rows: list[dict[str, Any]] = []
    for _ in range(400000):
        width = rng.choice((5, 7))
        items = rng.sample(range(1, 100), width)
        key = tuple(items)
        if key in seen or items == sorted(items):
            continue
        seen.add(key)
        rows.append({"items": items})
        if len(rows) == 8000:
            return rows
    raise RuntimeError("middle: could not draw 8000 distinct unsorted lists")


MIDDLE = Family(
    name="middle",
    instruction="Select one number from the list.",
    programs=("index_1", "middle", "median", "second_smallest"),
    render=lambda row: (
        "list = " + " ".join(str(item) for item in row["items"]) + "\nselect ="
    ),
    answer=_middle_answer,
    difficulty=lambda row: (len(row["items"]) - 3) // 2,
    inside=_middle_inside,
    outside=_middle_outside,
)


def _count_answer(program: str, row: dict[str, Any]) -> str:
    text = str(row["text"])
    letters = text.count("a")
    return str(
        {
            "count_a": letters,
            "parity_a": letters % 2,
            "presence_a": int(letters > 0),
            "count_vowels": sum(character in VOWELS for character in text),
        }[program]
    )


def _count_inside() -> list[dict[str, Any]]:
    rng = _family_rng("count", "inside")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for _ in range(400000):
        width = rng.randint(6, 14)
        letters = [rng.choice(CONSONANTS) for _ in range(width)]
        if rng.random() < 0.5:
            letters[rng.randrange(width)] = "a"
        text = "".join(letters)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text})
        if len(rows) == 20000:
            return rows
    raise RuntimeError("count: could not draw 20000 distinct low-vowel texts")


def _count_outside() -> list[dict[str, Any]]:
    rng = _family_rng("count", "outside")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for _ in range(400000):
        width = rng.randint(8, 16)
        letters = [rng.choice(CONSONANTS) for _ in range(width)]
        positions = rng.sample(range(width), rng.randint(3, 7))
        for offset, position in enumerate(positions):
            letters[position] = (
                "a" if offset < rng.randint(2, len(positions) - 1)
                else rng.choice("eiou")
            )
        text = "".join(letters)
        if text in seen or text.count("a") < 2:
            continue
        seen.add(text)
        rows.append({"text": text})
        if len(rows) == 8000:
            return rows
    raise RuntimeError("count: could not draw 8000 distinct multi-vowel texts")


COUNT = Family(
    name="count",
    instruction="Report the number the text calls for.",
    programs=("count_a", "parity_a", "presence_a", "count_vowels"),
    render=lambda row: f"text = {row['text']}\ncount =",
    answer=_count_answer,
    difficulty=lambda row: max(str(row["text"]).count("a") - 1, 0),
    inside=_count_inside,
    outside=_count_outside,
)


def _field_answer(program: str, row: dict[str, Any]) -> str:
    names = [str(name) for name, _ in row["fields"]]
    values = [str(value) for _, value in row["fields"]]
    if program == "named_code":
        return values[names.index("code")]
    if program == "position_2":
        return values[1]
    if program == "longest_value":
        return max(values, key=len)
    return max(values)


def _field_value(rng: random.Random, width: int, first: str) -> str:
    return first + "".join(rng.choice(CONSONANTS) for _ in range(width - 1))


def _field_record(
    rng: random.Random, code_at: int, longest_at: int, alpha_at: int
) -> dict[str, Any] | None:
    """One record with each cue placed on a named field.

    Placing the cues rather than sampling and filtering is what makes the
    separating pool large: four independent cues land on four different fields
    by chance about one record in ten, which is not enough to split into a
    diagnostic, a candidate pool and a probe set.
    """
    others = [name for name in FIELD_NAMES if name != "code"]
    names = rng.sample(others, 3)
    names.insert(code_at, "code")
    widths = rng.sample(range(3, 9), 4)
    firsts = rng.sample(CONSONANTS, 4)
    widths.sort()
    firsts.sort()
    order = [index for index in range(4) if index != longest_at]
    rng.shuffle(order)
    placed_widths = [0] * 4
    placed_widths[longest_at] = widths[-1]
    for slot, width in zip(order, widths[:-1], strict=True):
        placed_widths[slot] = width
    order = [index for index in range(4) if index != alpha_at]
    rng.shuffle(order)
    placed_firsts = [""] * 4
    placed_firsts[alpha_at] = firsts[-1]
    for slot, first in zip(order, firsts[:-1], strict=True):
        placed_firsts[slot] = first
    values = [
        _field_value(rng, placed_widths[index], placed_firsts[index])
        for index in range(4)
    ]
    if len(set(values)) != 4:
        return None
    return {"fields": [[names[index], values[index]] for index in range(4)]}


def _field_pool(mode: str, count: int) -> list[dict[str, Any]]:
    rng = _family_rng("field", mode)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for _ in range(400000):
        if mode == "inside":
            places = (1, 1, 1)
        elif mode == "separating":
            places = tuple(rng.sample((0, 2, 3), 3))
        else:
            places = tuple(rng.choices(range(4), k=3))
        record = _field_record(rng, *places)
        if record is None:
            continue
        key = json.dumps(record, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append(record)
        if len(rows) == count:
            return rows
    raise RuntimeError(f"field: could not draw {count} distinct {mode} records")


FIELD = Family(
    name="field",
    instruction="Report one value from the record.",
    programs=("named_code", "position_2", "longest_value", "alpha_last"),
    render=lambda row: (
        "record: "
        + " ".join(f"{name}={value}" for name, value in row["fields"])
        + "\nvalue ="
    ),
    answer=_field_answer,
    # Every separating record moves all three cues off the code field, so the
    # structural distance the other families use is constant here. Total value
    # length is the axis that does vary, and it is what a longer record costs.
    difficulty=lambda row: sum(len(str(value)) for _, value in row["fields"]),
    inside=lambda: _field_pool("inside", 20000),
    outside=lambda: _field_pool("separating", 5000) + _field_pool("partial", 3000),
)


FAMILIES: dict[str, Family] = {
    family.name: family for family in (CHAIN, MIDDLE, COUNT, FIELD)
}


# ---------------------------------------------------------------- input pools


@dataclass(frozen=True)
class Pools:
    """The three kinds of input a family produces, sorted by what they settle."""

    ambiguous: list[dict[str, Any]]
    separating: list[dict[str, Any]]
    partial: list[dict[str, Any]]


def answers(family: Family, row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(family.answer(program, row) for program in family.programs)


def build_pools(family: Family) -> Pools:
    """Partition the family's inputs by how many programs they tell apart.

    Fixed by the family alone, so every model and every seed sees the same
    universe and only the split into training, diagnostic and probe moves.
    """
    ambiguous = []
    for row in family.inside():
        if len(set(answers(family, row))) != 1:
            raise ValueError(
                f"{family.name}: an input inside the training region separates"
                " the programs"
            )
        ambiguous.append(row)
    separating: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for row in family.outside():
        distinct = len(set(answers(family, row)))
        if distinct == len(family.programs):
            separating.append(row)
        elif distinct > 1:
            partial.append(row)
    if len(separating) < 512 or len(partial) < 256:
        raise ValueError(
            f"{family.name}: {len(separating)} separating and {len(partial)}"
            " partial inputs are too few to split"
        )
    return Pools(ambiguous=ambiguous, separating=separating, partial=partial)


@dataclass(frozen=True)
class Split:
    """One seed's disjoint slices of a family's pools."""

    family: Family
    seed: int
    base: list[dict[str, Any]]
    extra: list[dict[str, Any]]
    ambiguous_eval: list[dict[str, Any]]
    diagnostic: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    probe: list[dict[str, Any]]


def split_pools(
    family: Family, pools: Pools, seed: int, config: dict[str, Any]
) -> Split:
    base_rows = int(config["base_rows"])
    extra_rows = max(int(value) for value in config["ambiguous_extra"])
    eval_rows = int(config["ambiguous_eval_rows"])
    diagnostic_rows = int(config["diagnostic_rows"])
    candidate_rows = int(config["targeted_candidates"])
    probe_rows = int(config["probe_rows"])

    rng = random.Random(f"{family.name}/{seed}")
    ambiguous = list(pools.ambiguous)
    rng.shuffle(ambiguous)
    needed = base_rows + extra_rows + eval_rows
    if len(ambiguous) < needed:
        raise ValueError(
            f"{family.name}: {len(ambiguous)} ambiguous inputs cannot cover"
            f" {needed} training and evaluation rows"
        )
    separating = list(pools.separating)
    rng.shuffle(separating)
    if len(separating) < diagnostic_rows + candidate_rows + probe_rows:
        raise ValueError(
            f"{family.name}: {len(separating)} separating inputs cannot cover"
            " a disjoint diagnostic, candidate and probe split"
        )
    first, second = base_rows, base_rows + extra_rows
    left, middle = diagnostic_rows, diagnostic_rows + candidate_rows
    return Split(
        family=family,
        seed=seed,
        base=ambiguous[:first],
        extra=ambiguous[first:second],
        ambiguous_eval=ambiguous[second : second + eval_rows],
        diagnostic=separating[:left],
        candidates=separating[left:middle],
        probe=separating[middle : middle + probe_rows],
    )


# ------------------------------------------------------------------- examples


def _row_id(family: Family, row: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return f"{family.name}-{digest}"


def make_example(
    family: Family,
    row: dict[str, Any],
    program: str,
    shots: Sequence[dict[str, Any]] = (),
) -> Example:
    """One prompt-response pair, labelled by `program`.

    `shots` prepends demonstrations from the training region, which is how the
    frozen base is asked the question at all. Trained arms use no shots, so the
    prompt they see at test time is the prompt they were trained on. Every
    demonstration is answered by the first program, which costs nothing because
    the training region is exactly where all four give the same answer -- and
    it makes the prompt identical for all four candidate answers, so a
    likelihood comparison across programs compares only the answers.
    """
    body = [family.instruction, ""]
    for shot in shots:
        body.append(family.render(shot))
        body.append(family.answer(family.programs[0], shot))
        body.append("")
    body.append(family.render(row))
    return Example(
        example_id=f"{_row_id(family, row)}-{program}-{len(shots)}",
        prompt="\n".join(body),
        response=" " + family.answer(program, row),
        metadata={
            "family": family.name,
            "program": program,
            "input": row,
            "shots": len(shots),
            "difficulty": family.difficulty(row),
        },
    )


def training_examples(
    family: Family, rows: Sequence[dict[str, Any]], program: str
) -> list[Example]:
    return [make_example(family, row, program) for row in rows]


# ---------------------------------------------------------------- measurement


def program_mass(
    session: ModelSession,
    family: Family,
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    shots: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Behavioural mass over the candidate programs, by answer likelihood.

    For every input each program's answer is scored under the model, and the
    four total log-likelihoods are turned into a distribution. Likelihood
    rather than generation, because a frozen base model that cannot produce the
    answer format at all still has a preference between the four answers, and
    that preference is what the targeting rule needs.
    """
    programs = family.programs
    examples = [
        make_example(family, row, program, shots)
        for row in rows
        for program in programs
    ]
    scores = completion_nll(
        session.model,
        session.tokenizer,
        examples,
        session.spec,
        int(config["max_length"]),
        int(config["scoring_batch_size"]),
    )
    per_row: list[dict[str, Any]] = []
    width = len(programs)
    for index, row in enumerate(rows):
        block = scores[index * width : (index + 1) * width]
        logits = torch.tensor([-score["sum_nll"] for score in block])
        probabilities = torch.softmax(logits, dim=-1).tolist()
        per_row.append(
            {
                "example_id": _row_id(family, row),
                "input": row,
                "sum_nll": {
                    program: block[position]["sum_nll"]
                    for position, program in enumerate(programs)
                },
                "mass": dict(zip(programs, probabilities, strict=True)),
                "argmax": max(
                    zip(programs, probabilities, strict=True), key=lambda pair: pair[1]
                )[0],
            }
        )
    mass = {
        program: statistics.fmean(row["mass"][program] for row in per_row)
        for program in programs
    }
    argmax_share = {
        program: sum(row["argmax"] == program for row in per_row) / len(per_row)
        for program in programs
    }
    return {"rows": per_row, "mass": mass, "argmax_share": argmax_share}


def _clean(text: str) -> str:
    return text.strip().split("\n")[0].strip()


def program_agreement(
    session: ModelSession,
    family: Family,
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    shots: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Exact agreement between greedy generation and each candidate program."""
    examples = [
        make_example(family, row, family.programs[0], shots) for row in rows
    ]
    records = generate_response_records(
        session.model,
        session.tokenizer,
        examples,
        session.spec,
        int(config["generation_batch_size"]),
        int(config["generation_max_new_tokens"]),
        generation_seed=0,
    )
    predictions = []
    for row, record in zip(rows, records, strict=True):
        text = _clean(str(record["response"]))
        matched = [
            program
            for program in family.programs
            if text == family.answer(program, row)
        ]
        predictions.append(
            {
                "example_id": _row_id(family, row),
                "response": text,
                "matches": matched,
            }
        )
    total = max(len(predictions), 1)
    agreement = {
        program: sum(program in row["matches"] for row in predictions) / total
        for program in family.programs
    }
    return {
        "rows": predictions,
        "agreement": agreement,
        "unmatched_fraction": sum(not row["matches"] for row in predictions) / total,
    }


def ambiguous_accuracy(
    session: ModelSession,
    family: Family,
    rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> float:
    """Fit on the training region, where all four programs give one answer."""
    result = program_agreement(session, family, rows, config)
    return max(result["agreement"].values()) if rows else 0.0


def gradient_influence(
    session: ModelSession,
    family: Family,
    rows: Sequence[dict[str, Any]],
    program: str,
    config: dict[str, Any],
) -> list[float]:
    """Gradient norm of each candidate example at the untrained adapter state.

    This is the third thing the targeted example could be special for, after
    difficulty and loss, so a control has to be matched on it. It is measured
    with the adapter attached and at its initial state, which is the state the
    example would actually act on.
    """
    from fineqcomp.modeling import CausalExampleDataset, causal_collate

    parameters = [
        parameter for parameter in session.model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("no trainable tensors, so influence cannot be measured")
    examples = [make_example(family, row, program) for row in rows]
    dataset = CausalExampleDataset(
        session.tokenizer, examples, session.spec, int(config["max_length"])
    )
    device = model_device(session.model)
    norms: list[float] = []
    was_training = session.model.training
    session.model.train()
    for index in range(len(dataset)):
        batch = causal_collate([dataset[index]], session.tokenizer.pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        session.model.zero_grad(set_to_none=True)
        loss = session.model(**batch).loss
        loss.backward()
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().pow(2).sum().item())
        norms.append(math.sqrt(total))
    session.model.zero_grad(set_to_none=True)
    if not was_training:
        session.model.eval()
    return norms


# --------------------------------------------------------------- the lock file


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def matched_control(
    table: list[dict[str, Any]], targeted: int, key: str, window: int
) -> dict[str, Any]:
    """The nearest candidates on one nuisance axis, then the least committed.

    Matching alone is not a control: half the candidate pool matches the
    targeted example on difficulty, and picking any of them would leave the
    comparison to chance. So the window is closed on `key` and the arm takes
    the candidate the base model is *least* committed to the dominant program
    on -- the opposite end of the axis the targeting rule reads. What separates
    the targeted arm from this one is the diagnostic and nothing else.
    """
    if window < 1:
        raise ValueError("the matched window must hold at least one candidate")
    value = float(table[targeted][key])
    others = [index for index in range(len(table)) if index != targeted]
    if not others:
        raise ValueError("no candidate is left to match against")
    nearest = sorted(others, key=lambda index: (abs(table[index][key] - value), index))
    chosen = min(
        nearest[:window],
        key=lambda index: (table[index]["mass_on_dominant"], index),
    )
    return {**table[chosen], "matched_on": key, "matched_value": value}


def lock_path(root: Path, model_key: str, family: str, seed: int) -> Path:
    return root / "locks" / model_key / f"{family}-seed{seed}.json"


def diagnose_cell(
    *,
    session: ModelSession,
    family: Family,
    pools: Pools,
    seed: int,
    config: dict[str, Any],
    out_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Measure the base model's version-space posterior and lock one example.

    The rule is fixed here rather than in the config: the dominant program is
    the argmax of the base model's mass, the target is the runner-up, and the
    targeted example is the separating input the base is most committed to the
    dominant program on. Nothing about it can be re-chosen once a training
    result exists, because the lock is written before any run does.
    """
    target = lock_path(out_root, session.spec.key, family.name, seed)
    if target.is_file() and not force:
        return json.loads(target.read_text())

    split = split_pools(family, pools, seed, config)
    shots = split.base[: int(config["diagnostic_shots"])]
    started = time.perf_counter()

    diagnostic = program_mass(session, family, split.diagnostic, config, shots)
    order = sorted(
        family.programs, key=lambda program: -diagnostic["mass"][program]
    )
    dominant, runner_up = order[0], order[1]

    candidates = program_mass(session, family, split.candidates, config, shots)
    losses = completion_nll(
        session.model,
        session.tokenizer,
        [make_example(family, row, runner_up) for row in split.candidates],
        session.spec,
        int(config["max_length"]),
        int(config["scoring_batch_size"]),
    )
    table = [
        {
            "input": row,
            "example_id": _row_id(family, row),
            "mass_on_dominant": candidates["rows"][index]["mass"][dominant],
            "difficulty": family.difficulty(row),
            "mean_nll": losses[index]["mean_nll"],
        }
        for index, row in enumerate(split.candidates)
    ]
    best = max(
        range(len(table)),
        key=lambda index: (table[index]["mass_on_dominant"], -index),
    )

    window = int(config["matched_window"])
    matched_difficulty = matched_control(table, best, "difficulty", window)
    matched_loss = matched_control(table, best, "mean_nll", window)
    random_pick = random.Random(f"random-arm/{family.name}/{session.spec.key}/{seed}")
    random_example = random_pick.choice(
        [row for index, row in enumerate(table) if index != best]
    )

    record = {
        "version": 1,
        "stage": "diagnose",
        "model": session.spec.key,
        "model_name": session.spec.name,
        "family": family.name,
        "seed": seed,
        "programs": list(family.programs),
        "diagnostic_shots": int(config["diagnostic_shots"]),
        "diagnostic_rows": len(split.diagnostic),
        "base_mass": diagnostic["mass"],
        "base_argmax_share": diagnostic["argmax_share"],
        "program_order": order,
        "dominant_program": dominant,
        "target_program": runner_up,
        "predicted_post_training_program": runner_up,
        "targeted": dict(table[best]),
        "controls": {
            "matched_difficulty": dict(matched_difficulty),
            "matched_loss": dict(matched_loss),
            "random_distinguishing": dict(random_example),
        },
        "matched_window": window,
        "candidate_table": table,
        "elapsed_seconds": time.perf_counter() - started,
        "finished_at": time.time(),
    }
    _write_json(target, record)
    _write_json(
        out_root / "diagnostics" / session.spec.key / f"{family.name}-seed{seed}.json",
        {"diagnostic": diagnostic["rows"], "candidates": candidates["rows"]},
    )
    return record


# ------------------------------------------------------------------- the arms


def arm_names(config: dict[str, Any]) -> list[str]:
    return [
        "base",
        "targeted",
        "matched_difficulty",
        "matched_loss",
        "matched_influence",
        "random_distinguishing",
        *[f"ambiguous_{int(count)}" for count in config["ambiguous_extra"]],
    ]


def build_arm(
    family: Family,
    split: Split,
    lock: dict[str, Any],
    arm: str,
    influence_input: dict[str, Any] | None,
) -> list[Example]:
    """The training rows for one arm: the shared base set plus its own addition."""
    program = str(lock["target_program"])
    rows = list(split.base)
    if arm == "base":
        pass
    elif arm == "targeted":
        rows.append(lock["targeted"]["input"])
    elif arm in {"matched_difficulty", "matched_loss", "random_distinguishing"}:
        rows.append(lock["controls"][arm]["input"])
    elif arm == "matched_influence":
        if influence_input is None:
            raise ValueError("the influence control needs a measured example")
        rows.append(influence_input)
    elif arm.startswith("ambiguous_"):
        count = int(arm.split("_")[1])
        if count > len(split.extra):
            raise ValueError(f"{arm}: only {len(split.extra)} spare ambiguous rows")
        rows.extend(split.extra[:count])
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return training_examples(family, rows, program)


def epochs_for_budget(rows: int, spec: TrainingSpec, updates: int) -> int:
    """Epochs that let an arm reach the update budget, which the cap then holds.

    Row counts across the arms differ by two orders of magnitude, so no single
    epoch count reaches one budget. The epochs are derived per arm and the cap
    in `TrainingSpec.max_updates` stops every one of them at the same update.
    """
    per_epoch = math.ceil(
        math.ceil(rows / spec.micro_batch_size)
        / (spec.effective_batch_size // spec.micro_batch_size)
    )
    return max(math.ceil(updates / max(per_epoch, 1)), 1)


# -------------------------------------------------------------------- a cell


@dataclass(frozen=True)
class Cell:
    family: str
    model: str
    arm: str
    seed: int

    @property
    def slug(self) -> str:
        return f"{self.family}/{self.model}/{self.arm}/seed{self.seed}"


def run_cell(
    *,
    config: dict[str, Any],
    campaign: dict[str, Any],
    session: ModelSession,
    pools: Pools,
    cell: Cell,
    out_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = out_root / "cells" / cell.family / cell.model / cell.arm / f"seed{cell.seed}"
    result_path = out_dir / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text())

    family = FAMILIES[cell.family]
    lock_file = lock_path(out_root, cell.model, cell.family, cell.seed)
    if not lock_file.is_file():
        raise FileNotFoundError(
            f"{lock_file} is missing; run the diagnose stage before training"
        )
    lock = json.loads(lock_file.read_text())
    if lock["model"] != cell.model or lock["family"] != cell.family:
        raise ValueError(f"{lock_file} does not describe {cell.slug}")

    split = split_pools(family, pools, cell.seed, config)
    adapter_spec = _adapter(str(config["adapter"]), campaign)
    training = TrainingSpec(**config["training"])
    updates = int(config["optimizer_updates"])
    started = time.perf_counter()

    session.attach(adapter_spec, cell.seed)
    try:
        influence_input = None
        influence_report: dict[str, Any] | None = None
        if cell.arm == "matched_influence":
            # Gradient norm is the one nuisance axis the diagnose stage cannot
            # measure: it depends on the adapter, and the adapter depends on
            # the seed. Every other control comes straight out of the lock.
            table = list(lock["candidate_table"])
            targeted_id = str(lock["targeted"]["example_id"])
            pool = [row for row in table if row["example_id"] != targeted_id]
            pool = pool[: int(config["influence_candidates"])]
            norms = gradient_influence(
                session,
                family,
                [lock["targeted"]["input"], *[row["input"] for row in pool]],
                str(lock["target_program"]),
                config,
            )
            scored = [
                {**row, "gradient_norm": norms[index + 1]}
                for index, row in enumerate(pool)
            ]
            scored.append({**lock["targeted"], "gradient_norm": norms[0]})
            chosen = matched_control(
                scored, len(scored) - 1, "gradient_norm", int(lock["matched_window"])
            )
            influence_input = chosen["input"]
            influence_report = {
                "targeted_gradient_norm": norms[0],
                "matched_gradient_norm": chosen["gradient_norm"],
                "candidates": len(pool),
            }

        examples = build_arm(family, split, lock, cell.arm, influence_input)
        spec = replace(
            training,
            epochs=epochs_for_budget(len(examples), training, updates),
            max_updates=updates,
        )
        metrics = train_adapter(
            session.model,
            session.tokenizer,
            examples,
            training_examples(
                family, split.ambiguous_eval[:64], str(lock["target_program"])
            ),
            session.spec,
            spec,
            cell.seed,
            out_dir / "training.jsonl",
        )
        if int(metrics["optimizer_updates"]) != updates:
            raise RuntimeError(
                f"{cell.slug} spent {metrics['optimizer_updates']} updates,"
                f" not {updates}"
            )

        agreement = program_agreement(session, family, split.probe, config)
        mass = program_mass(session, family, split.probe, config)
        fit = ambiguous_accuracy(session, family, split.ambiguous_eval, config)

        result = {
            "version": 1,
            "stage": "train",
            "family": cell.family,
            "model": cell.model,
            "arm": cell.arm,
            "seed": cell.seed,
            "programs": list(family.programs),
            "dominant_program": lock["dominant_program"],
            "target_program": lock["target_program"],
            "predicted_post_training_program": lock[
                "predicted_post_training_program"
            ],
            "train_rows": len(examples),
            "optimizer_updates": updates,
            "epochs": spec.epochs,
            "training": metrics,
            "influence": influence_report,
            "training_region_accuracy": fit,
            "probe_rows": len(split.probe),
            "agreement": agreement["agreement"],
            "unmatched_fraction": agreement["unmatched_fraction"],
            "mass": mass["mass"],
            "argmax_share": mass["argmax_share"],
            "elapsed_seconds": time.perf_counter() - started,
            "finished_at": time.time(),
        }
        _write_json(result_path, result)
        _write_json(out_dir / "probe_predictions.json", agreement["rows"])
        return result
    finally:
        session.unload()


# --------------------------------------------------------------------- grids


def expand_cells(config: dict[str, Any]) -> list[Cell]:
    cells = [
        Cell(family=str(family), model=str(model), arm=arm, seed=int(seed))
        for family in config["families"]
        for model in config["models"]
        for arm in arm_names(config)
        for seed in config["seeds"]
    ]
    unknown = {cell.family for cell in cells} - set(FAMILIES)
    if unknown:
        raise KeyError(f"unknown families: {sorted(unknown)}")
    return cells


def shard(items: list[Any], index: int, count: int) -> list[Any]:
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index} is outside 0-{count - 1}")
    return [item for position, item in enumerate(items) if position % count == index]


# ----------------------------------------------------------------- aggregate


def _cell_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "cells").glob("*/*/*/seed*/result.json")):
        record = json.loads(path.read_text())
        target = str(record["target_program"])
        dominant = str(record["dominant_program"])
        rows.append(
            {
                "family": record["family"],
                "model": record["model"],
                "arm": record["arm"],
                "seed": record["seed"],
                "dominant_program": dominant,
                "target_program": target,
                "train_rows": record["train_rows"],
                "training_region_accuracy": record["training_region_accuracy"],
                "agreement_target": record["agreement"][target],
                "agreement_dominant": record["agreement"][dominant],
                "unmatched_fraction": record["unmatched_fraction"],
                "mass_target": record["mass"][target],
                "mass_dominant": record["mass"][dominant],
                "argmax_program": max(
                    record["agreement"], key=lambda key: record["agreement"][key]
                ),
                "elapsed_seconds": record["elapsed_seconds"],
            }
        )
    return rows


def _by_arm(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, dict]]:
    grouped: dict[tuple[str, str, int], dict[str, dict]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["model"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["arm"])] = row
    return grouped


def check_gates(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered decision rule, one clause at a time.

    G1  every arm still fits the training region, so a program read off the
        probes is a program the arm actually learned.
    G2  ambiguous data does not move the dominant program.
    G3  the targeted example does: agreement with the predicted program rises
        by at least the registered margin and it becomes the argmax.
    G4  none of the matched or random controls reproduces that switch.
    G5  G3 and G4 hold on a majority of seeds for the same family and model.
    """
    gates = config["gates"]
    grouped = _by_arm(rows)
    ambiguous_arms = [f"ambiguous_{int(count)}" for count in config["ambiguous_extra"]]
    controls = [
        "matched_difficulty",
        "matched_loss",
        "matched_influence",
        "random_distinguishing",
    ]
    per_cell = []
    for (family, model, seed), arms in sorted(grouped.items()):
        if "base" not in arms or "targeted" not in arms:
            continue
        base = arms["base"]
        targeted = arms["targeted"]
        fit = min(row["training_region_accuracy"] for row in arms.values())
        drift = [
            abs(arms[arm]["agreement_dominant"] - base["agreement_dominant"])
            for arm in ambiguous_arms
            if arm in arms
        ]
        control_gain = [
            arms[arm]["agreement_target"] - base["agreement_target"]
            for arm in controls
            if arm in arms
        ]
        gain = targeted["agreement_target"] - base["agreement_target"]
        per_cell.append(
            {
                "family": family,
                "model": model,
                "seed": seed,
                "arms": len(arms),
                "g1_min_training_region_accuracy": fit,
                "g1": fit >= float(gates["training_region_accuracy"]),
                "g2_max_ambiguous_drift": max(drift) if drift else None,
                "g2": bool(drift)
                and max(drift) <= float(gates["ambiguous_stability"]),
                "g3_targeted_gain": gain,
                "g3": gain >= float(gates["targeted_switch_gain"])
                and targeted["argmax_program"] == targeted["target_program"],
                "g4_best_control_gain": max(control_gain) if control_gain else None,
                "g4": bool(control_gain)
                and gain - max(control_gain) >= float(gates["control_margin"]),
            }
        )
    replication = []
    keys = sorted({(row["family"], row["model"]) for row in per_cell})
    for family, model in keys:
        members = [
            row
            for row in per_cell
            if row["family"] == family and row["model"] == model
        ]
        passed = [row for row in members if row["g3"] and row["g4"]]
        replication.append(
            {
                "family": family,
                "model": model,
                "seeds": len(members),
                "seeds_passing_g3_g4": len(passed),
                "g5": len(members) > 1 and len(passed) * 2 > len(members),
            }
        )
    return {
        "cells": per_cell,
        "replication": replication,
        "passed": bool(per_cell)
        and all(row["g1"] for row in per_cell)
        and all(row["g2"] for row in per_cell)
        and any(row["g5"] for row in replication),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    rows = _cell_rows(root)
    gates = check_gates(rows, config)
    _write_csv(root / "arms.csv", rows)
    _write_csv(root / "gates.csv", gates["cells"])
    _write_csv(root / "replication.csv", gates["replication"])
    summary = {"cells": len(rows), "gates": gates}
    _write_json(root / "summary.json", summary)
    return summary


# ----------------------------------------------------------------------- cli


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or int(raw.get("version", 0)) != 1:
        raise ValueError(f"{path}: expected program-selection config version 1")
    missing = {
        "models",
        "adapter",
        "families",
        "seeds",
        "base_rows",
        "ambiguous_extra",
        "optimizer_updates",
        "training",
        "gates",
    } - set(raw)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    return {
        # Scoring widths live next to the study rather than in the training
        # spec, because they only ever describe how a finished model is read.
        "scoring_batch_size": 16,
        "generation_batch_size": 32,
        "generation_max_new_tokens": 12,
        "influence_candidates": 96,
        "matched_window": 32,
        "diagnostic_rows": 512,
        "diagnostic_shots": 8,
        "targeted_candidates": 512,
        "probe_rows": 1024,
        "ambiguous_eval_rows": 256,
        **raw,
        "max_length": int(raw["training"]["max_length"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-specific program selection under fine-tuning."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/program_selection.yaml")
    )
    parser.add_argument(
        "--campaign", type=Path, default=Path("configs/transposed_receiver_panel.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("runs_program_selection"))
    parser.add_argument("--stage", choices=("diagnose", "train"), default="train")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--audit", action="store_true", help="check the pools only")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero if a cell misses the training-region gate",
    )
    return parser


def selected_cells(config: dict[str, Any], args: argparse.Namespace) -> list[Cell]:
    cells = expand_cells(config)
    if args.models:
        cells = [cell for cell in cells if cell.model in set(args.models)]
    if args.families:
        cells = [cell for cell in cells if cell.family in set(args.families)]
    if args.arms:
        cells = [cell for cell in cells if cell.arm in set(args.arms)]
    if args.seeds:
        cells = [cell for cell in cells if cell.seed in set(args.seeds)]
    return shard(cells, args.shard, args.shards)


def audit_pools(config: dict[str, Any]) -> dict[str, Any]:
    """Prove each family is ambiguous where it claims to be, without a GPU."""
    report = []
    for name in config["families"]:
        family = FAMILIES[str(name)]
        pools = build_pools(family)
        splits = [
            split_pools(family, pools, int(seed), config) for seed in config["seeds"]
        ]
        overlaps = 0
        for split in splits:
            slices = [split.base, split.extra, split.ambiguous_eval]
            keys = [
                {_row_id(family, row) for row in part} for part in slices
            ]
            overlaps += len(keys[0] & keys[1]) + len(keys[0] & keys[2])
            probe = {_row_id(family, row) for row in split.probe}
            diagnostic = {_row_id(family, row) for row in split.diagnostic}
            candidates = {_row_id(family, row) for row in split.candidates}
            overlaps += len(probe & diagnostic) + len(probe & candidates)
            overlaps += len(diagnostic & candidates)
        report.append(
            {
                "family": family.name,
                "programs": list(family.programs),
                "ambiguous": len(pools.ambiguous),
                "separating": len(pools.separating),
                "partial": len(pools.partial),
                "split_overlaps": overlaps,
                "example_prompt": make_example(
                    family, pools.separating[0], family.programs[0]
                ).prompt,
                "example_answers": {
                    program: family.answer(program, pools.separating[0])
                    for program in family.programs
                },
                "passed": overlaps == 0,
            }
        )
    return {"families": report, "passed": all(row["passed"] for row in report)}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.audit:
        print(json.dumps(audit_pools(config), indent=2, sort_keys=True))
        return
    if args.aggregate:
        print(json.dumps(aggregate(args.out, config), indent=2, sort_keys=True))
        return

    campaign = load_campaign(args.campaign)
    cells = selected_cells(config, args)
    if args.stage == "diagnose":
        work = sorted({(cell.model, cell.family, cell.seed) for cell in cells})
    else:
        work = cells
    if args.dry_run:
        for item in work:
            print(item if isinstance(item, tuple) else item.slug)
        print(f"{len(work)} units")
        return
    if not work:
        raise SystemExit("shard selected no work")

    pools = {name: build_pools(FAMILIES[str(name)]) for name in config["families"]}
    models = sorted({cell.model for cell in cells})
    failed: list[str] = []
    for model_key in models:
        session = ModelSession.load(_model(model_key, campaign))
        try:
            if args.stage == "diagnose":
                for key, family_name, seed in [
                    item for item in work if item[0] == model_key
                ]:
                    record = diagnose_cell(
                        session=session,
                        family=FAMILIES[family_name],
                        pools=pools[family_name],
                        seed=seed,
                        config=config,
                        out_root=args.out,
                        force=args.force,
                    )
                    print(
                        json.dumps(
                            {
                                "model": key,
                                "family": family_name,
                                "seed": seed,
                                "dominant": record["dominant_program"],
                                "target": record["target_program"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            else:
                for cell in [item for item in work if item.model == model_key]:
                    result = run_cell(
                        config=config,
                        campaign=campaign,
                        session=session,
                        pools=pools[cell.family],
                        cell=cell,
                        out_root=args.out,
                        force=args.force,
                    )
                    if float(result["training_region_accuracy"]) < float(
                        config["gates"]["training_region_accuracy"]
                    ):
                        failed.append(cell.slug)
                    print(
                        json.dumps(
                            {
                                "cell": cell.slug,
                                "target_agreement": round(
                                    float(
                                        result["agreement"][result["target_program"]]
                                    ),
                                    4,
                                ),
                                "seconds": round(
                                    float(result["elapsed_seconds"]), 1
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        finally:
            try:
                session.unload()
            except Exception:
                pass
            del session
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # The smoke runs with --gate so a grid of 360 cells never starts behind a
    # task nothing can learn. G1 is the only gate a single shard can read; the
    # rest need the whole grid and are checked by --aggregate.
    if args.gate and failed:
        raise SystemExit(
            f"{len(failed)} cells missed the training-region gate: "
            + ", ".join(sorted(failed))
        )


if __name__ == "__main__":
    main()
