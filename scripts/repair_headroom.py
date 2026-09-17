#!/usr/bin/env python3
"""Is repairing corrupted work gated by how much headroom a model has?

Belongs to the reasoning-trajectory line, not to fineQComp; it lives here only
because this checkout already holds the prepared GSM8K and GSM-Symbolic splits.

The prior work (Tyen et al., BIG-Bench Mistake) asks whether models can find
injected mistakes in a chain of thought, on a fixed task set. The question here
is different and relational: **repair should only be available where the model
already has competence to spare.** So difficulty is calibrated per model rather
than fixed -- every model is scored on its own solo accuracy over the same
items, and the repair rate is read against *that*, not against an absolute
notion of hard.

Four conditions per item, identical text across models:

  solo              the question alone, sampled several times. This is the
                    x-axis: the model's own competence on this item.
  clean             the question plus a correct worked solution with the final
                    answer withheld. The control that says whether supplied work
                    is usable at all.
  corrupt_relevant  one intermediate value that the answer depends on is wrong.
  corrupt_distract  an appended line computes something the answer never uses,
                    and *that* is wrong. Corrupting work nobody reads should
                    cost nothing; this separates "notices any anomaly" from
                    "tracks the dependency structure".

Each work-supplied condition asks the model to flag an error if it finds one, so
the run measures detection rate and its false-positive rate on `clean` as well
as end accuracy. A model that cries error on clean work is not detecting
anything.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

ANNOTATION = re.compile(r"<<([^=<>]+)=([\-0-9.,]+)>>")
# The result of a step is the number after its LAST '=' -- true for the
# annotation-stripped GSM8K line "Janet sells 16 - 3 - 4 = 9 duck eggs a day."
# and for the GSM-Symbolic line "... is 27 + 9 = 36." alike. Matching on the
# span rather than the digits is what keeps a rewrite off the step's inputs:
# replacing the text "24" in ".24 * 100% = 24%" would corrupt both.
RESULT = re.compile(r"=\s*\$?(-?[\d,]+(?:\.\d+)?)")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
FINAL = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")


def as_number(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def gold_answer(answer: str) -> float | None:
    found = FINAL.search(answer)
    return as_number(found.group(1)) if found else None


def extract_steps(body: str) -> list[dict[str, Any]]:
    """Every line that states an intermediate result, with that result's span.

    `body` must already have calculator annotations stripped, so that one line
    holds one statement of the result rather than two.
    """
    steps = []
    for index, line in enumerate(body.split("\n")):
        if not line.strip():
            continue
        matches = list(RESULT.finditer(line))
        if not matches:
            continue
        last = matches[-1]
        value = as_number(last.group(1))
        if value is not None:
            steps.append({"line": index, "value": value, "span": last.span(1)})
    return steps


def perturb(value: float, rng: random.Random) -> float:
    """A wrong value that stays plausible: same sign, same rough magnitude.

    A corruption to 0, to a negative, or to something wildly off scale would be
    detectable on surface form alone, which is not the capability under test.
    """
    for _ in range(20):
        if abs(value) >= 10:
            delta = rng.choice([-1, 1]) * max(1, round(abs(value) * rng.uniform(0.1, 0.4)))
        else:
            delta = rng.choice([-3, -2, -1, 1, 2, 3])
        candidate = value + delta
        if candidate != value and (value <= 0 or candidate > 0):
            return round(candidate, 2) if isinstance(value, float) and value % 1 else candidate
    return value + 1


def format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def rewrite_result(line: str, span: tuple[int, int], new: float) -> str:
    """Rewrite exactly the result's characters, leaving the inputs untouched."""
    start, end = span
    return line[:start] + format_number(new) + line[end:]


def strip_annotations(text: str) -> str:
    return ANNOTATION.sub(lambda m: "", text)


def build_item(record: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    question, answer = record["question"], record["answer"]
    gold = gold_answer(answer)
    body = strip_annotations(answer.split("####")[0]).rstrip()
    steps = extract_steps(body)
    # At least two steps: the work is cut off *at* the corrupted one, so there
    # has to be something left for the model to compute afterwards.
    if gold is None or len(steps) < 2:
        return None
    lines = body.split("\n")

    # Corrupt the second-to-last step and truncate there. Leaving the later
    # lines in place would leave the correct final answer sitting in the work,
    # and a model could copy it without ever using the corrupted value -- the
    # first version of this did exactly that and measured nothing. Truncating
    # forces the remaining computation through the corrupted number.
    target = steps[-2]
    wrong = perturb(target["value"], rng)
    kept = lines[: target["line"] + 1]
    corrupted = list(kept)
    corrupted[target["line"]] = rewrite_result(
        corrupted[target["line"]], target["span"], wrong)

    numbers = [as_number(n) for n in NUMBER.findall(question)]
    numbers = [n for n in numbers if n is not None][:2]
    if len(numbers) < 2:
        return None
    total = numbers[0] + numbers[1]
    reference = "For reference, {} + {} = {}."
    distract_ok = reference.format(format_number(numbers[0]),
                                   format_number(numbers[1]), format_number(total))
    distract_bad = reference.format(format_number(numbers[0]),
                                    format_number(numbers[1]),
                                    format_number(perturb(total, rng)))
    clean_work = "\n".join(kept).strip()
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "question": question,
        "gold": gold,
        "corrupted_value": wrong,
        "original_value": target["value"],
        "steps_supplied": len(kept),
        "steps_total": len(lines),
        "work": {
            "clean": clean_work + "\n" + distract_ok,
            "corrupt_relevant": "\n".join(corrupted).strip() + "\n" + distract_ok,
            "corrupt_distract": clean_work + "\n" + distract_bad,
        },
    }


SOLO = ("Solve the problem. Show concise work and finish with '#### ' followed by "
        "the final numeric answer.\n\nQuestion: {question}")

# Deliberately not "this may or may not contain an error". The first version
# said that, and the 3B answered ERROR on 100% of clean work -- the suggestion
# alone was enough. The task is stated as continuation, with error reporting
# available but not invited, so that flagging reflects something the model
# noticed rather than something the prompt proposed. The clean condition
# measures whatever priming remains.
WORKED = ("Here is a problem and an unfinished solution written by someone else."
          "\n\nQuestion: {question}"
          "\n\nSolution so far:\n{work}\n\nContinue from where it stops and give "
          "the final answer. If while doing so you find that a line you were given "
          "is wrong, begin your reply with 'ERROR' and name the wrong line; "
          "otherwise begin with 'OK'. Finish with '#### ' followed by the correct "
          "final numeric answer.")


def prompts_for(item: dict[str, Any]) -> list[tuple[str, str]]:
    out = [("solo", SOLO.format(question=item["question"]))]
    for name, work in item["work"].items():
        out.append((name, WORKED.format(question=item["question"], work=work)))
    return out


def score(text: str, gold: float) -> dict[str, Any]:
    found = FINAL.search(text)
    value = as_number(found.group(1)) if found else None
    if value is None:
        numbers = NUMBER.findall(text)
        value = as_number(numbers[-1]) if numbers else None
    head = text.strip()[:200].upper()
    return {
        "answer": value,
        "correct": value is not None and abs(value - gold) < 1e-6,
        "flagged": head.startswith("ERROR") or "ERROR AT" in head,
    }


def load_items(root: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    gsm = root / "prepared/natural/cot_math_permuted/seed11/test.jsonl"
    if gsm.exists():
        for line in gsm.read_text().splitlines():
            row = json.loads(line)
            question = row["prompt"].split("Question:", 1)[-1].rsplit("Answer:", 1)[0].strip()
            records.append({"id": row["example_id"], "dataset": "gsm8k",
                            "question": question, "answer": row["response"]})
    sym = root / "prepared/gsm-symbolic-source.jsonl"
    if sym.exists():
        for line in sym.read_text().splitlines():
            row = json.loads(line)
            if row["instance"] != 0:
                continue
            records.append({"id": f"symbolic-{row['original_id']}", "dataset": "gsm_symbolic",
                            "question": row["question"], "answer": row["answer"]})
    items: list[dict[str, Any]] = []
    for dataset in ("gsm8k", "gsm_symbolic"):
        pool = [r for r in records if r["dataset"] == dataset]
        rng.shuffle(pool)
        built = []
        for record in pool:
            made = build_item(record, rng)
            if made:
                built.append(made)
            if len(built) >= limit:
                break
        items.extend(built)
    return items


def resolve_model(model: str) -> str:
    """Turn a repo id into the local snapshot directory holding its weights.

    The caches here were populated by tools that skip files like LICENSE and
    README, so `snapshot_download` calls them incomplete and refuses to serve
    them offline even though every weight shard is present. Pointing vLLM and
    the tokenizer straight at the snapshot directory sidesteps that check
    without enabling any network traffic -- which matters on a shared machine
    where an unexpected download is somebody else's bandwidth.
    """
    if Path(model).is_dir():
        return model
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    stem = "models--" + model.replace("/", "--")
    for hub in (home / "hub", home):
        for snapshot in sorted((hub / stem / "snapshots").glob("*")):
            if (snapshot / "config.json").exists():
                return str(snapshot)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--items", type=int, default=100)
    parser.add_argument("--solo-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.87)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="let Qwen3 models think before answering; the "
                             "default disables it so every rung answers under "
                             "the same budget")
    parser.add_argument("--dump-items", action="store_true",
                        help="build the items, write them, and exit without a GPU")
    args = parser.parse_args()

    items = load_items(args.root, args.items, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "items.json").write_text(json.dumps(items, indent=1))
    print(f"built {len(items)} items "
          f"({sum(i['dataset'] == 'gsm8k' for i in items)} gsm8k, "
          f"{sum(i['dataset'] == 'gsm_symbolic' for i in items)} symbolic)", flush=True)
    if args.dump_items:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    source = resolve_model(args.model)
    print(f"model source: {source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(source)

    def chat(text: str) -> str:
        messages = [{"role": "user", "content": text}]
        try:
            # Qwen3 exposes a thinking mode; Qwen2.5 does not. Disabling it by
            # default keeps inference compute comparable across the ladder, but
            # that turns "Qwen3 never flags an error" into a claim about Qwen3
            # *with thinking off* -- which is a different claim. Running the
            # same items with it on is how the two are told apart, so it is a
            # switch rather than a constant.
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking)
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

    jobs = []
    for item in items:
        for condition, prompt in prompts_for(item):
            count = args.solo_samples if condition == "solo" else 1
            for sample in range(count):
                jobs.append({"id": item["id"], "dataset": item["dataset"],
                             "condition": condition, "sample": sample,
                             "gold": item["gold"], "prompt": chat(prompt)})
    print(f"{len(jobs)} generations queued", flush=True)

    llm = LLM(model=source, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, enforce_eager=False,
              trust_remote_code=True, seed=args.seed)
    # Greedy for the single-shot conditions, sampled for solo: solo needs a pass
    # rate per item to place the model on the competence axis, and a greedy
    # decode gives only 0 or 1.
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    sampled = SamplingParams(temperature=0.7, top_p=0.95, seed=args.seed,
                             max_tokens=args.max_new_tokens)
    outputs = llm.generate([j["prompt"] for j in jobs],
                           [greedy if j["condition"] != "solo" else sampled
                            for j in jobs])
    rows = []
    for job, output in zip(jobs, outputs):
        text = output.outputs[0].text
        rows.append({**{k: job[k] for k in ("id", "dataset", "condition", "sample", "gold")},
                     **score(text, job["gold"]),
                     "completion_tokens": len(output.outputs[0].token_ids),
                     "text": text[:1500]})
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {len(rows)} rows to {args.out/'rows.json'}", flush=True)


if __name__ == "__main__":
    main()
