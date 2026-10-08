"""Functional spectrum of archived adapters: adapter, base model and training data together.

For every run in a cells CSV (run_id, model_key, dataset_key, seed), load the
run's frozen base model once per receiver, attach the archived adapter, and run
`functional_profile` on the first ROWS rows of the prepared training split that
adapter was trained on. Runs are dealt out across workers.

$STORE is not mounted on GPU nodes, so the adapters are first unpacked on
prepost to ADAPTERS/<run_id>/raw_channel.pt (on $SCRATCH). Then, on a GPU node:
    python functional.py CELLS ADAPTERS RUNS_ROOT PREPARED_ROOT OUT_PREFIX SHARD SHARDS [MODE ROWS SPLIT GAUGES]
MODE is `functional` (activation-weighted spectrum, the default) or `gain`
(the likelihood gain lost when each singular direction is removed), or
`attenuation` (five scales of the uncompressed update, 64 sampled rows).
`independent_rate` validates the full codec ladder on calibration rows left
after excluding the same ROWS sampled examples. It supplies targets, not features.
`gauge` reruns the run's codec ladder on the first ROWS calibration rows, the
rows its recorded R* was scored on, once per reparameterization (B Q^-1, Q A)
named in GAUGES, e.g. "identity orthogonal-d0 conditioned-k100-d1" (default:
identity and one conditioned draw). Rows already in the output are kept.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import torch

from fineqcomp.adapters import apply_adapter_tensors
from fineqcomp.config import RunSpec
from fineqcomp.data import read_jsonl
from fineqcomp.studies.generalisation import attenuation_profile, functional_profile, gain_profile
from fineqcomp.modeling import ModelSession
from fineqcomp.studies.relative_info import sample_examples

ROWS = 128


def _loss(session, update, rows, spec):
    from fineqcomp.training import causal_nll

    apply_adapter_tensors(session.model, update)
    # The run's own batching, so the identity factorization reproduces the
    # runner's ladder rather than differing by padding round-off.
    with torch.no_grad():
        return causal_nll(session.model, session.tokenizer, rows, spec.model,
                          spec.training.max_length, spec.training.micro_batch_size)["bits_per_token"]


def _base_loss(session, tensors, rows, spec):
    return _loss(session, {n: torch.zeros_like(t) if ".lora_B." in n else t
                           for n, t in tensors.items()}, rows, spec)


def codec_ladder(session, tensors, rows, spec, base):
    """The run's own codec ladder on `tensors`: raw gain, coded points and R*."""
    from fineqcomp.codec import decode_adapter_tensor_map, encode_tensor_map
    from fineqcomp.rstar import r_star

    points = []
    raw_gain = base - _loss(session, tensors, rows, spec)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "adapter.fqcb"
        for codec in spec.codecs:
            if codec.method != "uniform":
                raise ValueError("this validation expects the uniform codec ladder")
            metadata = {"run_id": spec.run_id, "adapter_method": spec.adapter.method,
                        "adapter_seed": spec.seed, "adapter_key": spec.adapter.key,
                        "codec_key": codec.key, "codec_method": codec.method}
            storage = encode_tensor_map(tensors, path, codec.bits, codec.quantizer,
                                        metadata=metadata, blend=codec.blend)
            _, decoded = decode_adapter_tensor_map(path)
            point = {"codec": codec.key, "effective_bits_per_value": storage["effective_bits_per_value"],
                     "relative_rmse": storage["relative_rmse"],
                     "heldout_bits_saved_per_token": base - _loss(session, decoded, rows, spec)}
            points.append(point)
            print(spec.run_id, json.dumps(point), flush=True)
    reference = max([raw_gain] + [p["heldout_bits_saved_per_token"] for p in points])
    best = r_star(points, reference=reference)
    return {"raw_gain": raw_gain, "r_star": best["r_star"], "bracketed": best["bracketed"],
            "r_star_raw": r_star(points, reference=raw_gain)["r_star"], "points": points}


def independent_rate_profile(session, tensors, rows, spec):
    """Measure validation targets on rows excluded from the attenuation probe."""
    try:
        ladder = codec_ladder(session, tensors, rows, spec, _base_loss(session, tensors, rows, spec))
        return {"independent_rows": len(rows), "independent_raw_gain": ladder["raw_gain"],
                "independent_r_star": ladder["r_star"],
                "independent_r_star_raw": ladder["r_star_raw"],
                "independent_points": json.dumps(ladder["points"]),
                "independent_example_ids": json.dumps([r.example_id for r in rows])}
    finally:
        apply_adapter_tensors(session.model, tensors)


def parse_gauge(token):
    """`conditioned-k100-d2` -> ("conditioned", 100.0, 2); kappa and draw default to 1 and 0."""
    kind, *options = token.split("-")
    values = {"k": 1.0, "d": 0}
    for option in options:
        if option[:1] not in values or len(option) < 2:
            raise ValueError(f"bad gauge option {option!r} in {token!r}")
        values[option[0]] = float(option[1:]) if option[0] == "k" else int(option[1:])
    return kind, values["k"], values["d"]


def gauge_profile(session, tensors, rows, spec, token, base):
    """R* of one adapter after the reparameterization (B Q^-1, Q A) named by `token`.

    `base` is the frozen model's loss on `rows`, shared by every gauge. The
    row also records the uncoded update's gain and weight drift, which must
    match the identity's, and the one-bit projection gamma that the rate rule
    divides by.
    """
    from fineqcomp.adapters import frobenius_inner, lora_pairs
    from fineqcomp.codec import binary_projection, regauge_lora

    kind, kappa, draw = parse_gauge(token)
    regauged = regauge_lora(tensors, kind, draw=draw, kappa=kappa)
    drift = norm = 0.0
    for a_name, b_name in lora_pairs(tensors):
        a, b = tensors[a_name].double(), tensors[b_name].double()
        a2, b2 = regauged[a_name].double(), regauged[b_name].double()
        norm += frobenius_inner(a, b, a, b)
        drift += frobenius_inner(a2, b2, a2, b2) + frobenius_inner(a, b, a, b) - 2 * frobenius_inner(a, b, a2, b2)
    try:
        ladder = codec_ladder(session, regauged, rows, spec, base)
    finally:
        apply_adapter_tensors(session.model, tensors)
    projection, residual = binary_projection(regauged)
    return {"gauge": token, "gauge_kind": kind, "gauge_kappa": kappa, "gauge_draw": draw,
            "gauge_rows": len(rows), "base_loss": base, "raw_gain": ladder["raw_gain"],
            "r_star": ladder["r_star"], "bracketed": ladder["bracketed"],
            "r_star_raw": ladder["r_star_raw"], "binary_projection": projection,
            "binary_residual": residual, "update_drift": max(drift, 0.0) ** 0.5 / norm ** 0.5,
            "points": json.dumps(ladder["points"])}


def main() -> None:
    cells, adapters, runs, prepared, out, shard, shards = sys.argv[1:8]
    shard, shards = int(shard), int(shards)
    mode = sys.argv[8] if len(sys.argv) > 8 else "functional"
    if mode not in {"functional", "gain", "attenuation", "independent_rate", "gauge"}:
        raise ValueError(f"unknown profile: {mode}")
    default_rows = {"attenuation": 64, "independent_rate": 64, "gauge": 256}.get(mode, ROWS)
    row_count = int(sys.argv[9]) if len(sys.argv) > 9 else default_rows
    if row_count < 1:
        raise ValueError("rows must be positive")
    split = sys.argv[10] if len(sys.argv) > 10 else "train"
    if split not in {"train", "calibration"}:
        raise ValueError(f"unknown split: {split}")
    gauges = (sys.argv[11] if len(sys.argv) > 11 else "identity conditioned-k10-d0").split()
    for token in gauges:
        parse_gauge(token)
    table = sorted(csv.DictReader(open(cells)), key=lambda r: (r["model_key"], r["run_id"]))
    # Runs, not receivers, are dealt out, so one large receiver spreads over
    # every worker. A receiver's runs can differ in adapter rank, so the session
    # is re-attached whenever the adapter spec changes.
    mine = table[shard::shards]
    output = Path(f"{out}_{shard}.csv")
    results = []
    if mode == "gauge" and output.is_file():
        # A gauge sweep resumes: rows already written are kept, not rerun.
        results = list(csv.DictReader(output.open()))
    done = {(r["run_id"], r.get("gauge")) for r in results}
    if mode == "gauge":
        mine = [r for r in mine if any((r["run_id"], g) not in done for g in gauges)]
    specs = [RunSpec.from_dict(json.loads((Path(runs) / r["run_id"] / "config.json").read_text()))
             for r in mine]

    def record(profile):
        results.append(profile)
        # Each completed adapter (or gauge) survives a later timeout or load failure.
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)

    session, attached = None, None
    try:
        for row, spec in sorted(zip(mine, specs), key=lambda x: (x[0]["model_key"], x[1].adapter.rank)):
            if session is None or attached[0] != spec.model:
                if session is not None:
                    session.unload()
                session, attached = ModelSession.load(spec.model), (spec.model, None)
            if attached[1] != spec.adapter:
                if attached[1] is not None:
                    session.unload()
                session.attach(spec.adapter, spec.seed)
                attached = (spec.model, spec.adapter)
            tensors = torch.load(Path(adapters) / row["run_id"] / "raw_channel.pt",
                                 map_location="cpu", weights_only=True)
            apply_adapter_tensors(session.model, tensors)
            data = read_jsonl(Path(prepared) / row["dataset_key"] / f"seed{row['seed']}" / f"{split}.jsonl")
            if mode == "gauge":
                if split != "calibration":
                    raise ValueError("the gauge sweep scores the calibration split")
                data = data[:row_count]
                base = _base_loss(session, tensors, data, spec)
                apply_adapter_tensors(session.model, tensors)
                for token in gauges:
                    if (row["run_id"], token) not in done:
                        record({"run_id": row["run_id"],
                                **gauge_profile(session, tensors, data, spec, token, base)})
                continue
            if mode == "independent_rate":
                if split != "calibration":
                    raise ValueError("independent validation requires the calibration split")
                probe = sample_examples(data, row_count, seed=271828)
                keys = {(r.prompt, r.response) for r in probe}
                data = [r for r in data if (r.prompt, r.response) not in keys]
                if not data:
                    raise ValueError("no examples remain after excluding the probe")
                profile = independent_rate_profile(session, tensors, data, spec)
                profile["probe_example_ids"] = json.dumps([r.example_id for r in probe])
            elif mode == "attenuation":
                data = sample_examples(data, row_count, seed=271828)
                profile = attenuation_profile(session, tensors, data, spec.model, spec.training.max_length)
                profile["attenuation_split"] = split
            elif mode == "gain":
                profile = gain_profile(session, tensors, data[:row_count], spec.model, spec.training.max_length)
            else:
                profile = functional_profile(session, data[:row_count], spec.model, spec.training.max_length)
            record({"run_id": row["run_id"], **profile})
            print(row["run_id"], json.dumps(profile), flush=True)
    finally:
        if session is not None:
            session.unload()


if __name__ == "__main__":
    main()
