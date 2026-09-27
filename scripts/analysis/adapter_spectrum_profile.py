"""Functional spectrum of archived adapters: adapter, base model and training data together.

For every run in a cells CSV (run_id, model_key, dataset_key, seed), load the
run's frozen base model once per receiver, attach the archived adapter, and run
`functional_profile` on the first ROWS rows of the prepared training split that
adapter was trained on. Runs are dealt out across workers.

$STORE is not mounted on GPU nodes, so the adapters are first unpacked on
prepost to ADAPTERS/<run_id>/raw_channel.pt (on $SCRATCH). Then, on a GPU node:
    python functional.py CELLS ADAPTERS RUNS_ROOT PREPARED_ROOT OUT_PREFIX SHARD SHARDS [MODE ROWS SPLIT]
MODE is `functional` (activation-weighted spectrum, the default) or `gain`
(the likelihood gain lost when each singular direction is removed), or
`attenuation` (five scales of the uncompressed update, 64 sampled rows).
`independent_rate` validates the full codec ladder on calibration rows left
after excluding the same ROWS sampled examples. It supplies targets, not features.
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


def independent_rate_profile(session, tensors, rows, spec):
    """Measure validation targets on rows excluded from the attenuation probe."""
    from fineqcomp.codec import decode_adapter_tensor_map, encode_tensor_map
    from fineqcomp.rstar import r_star
    from fineqcomp.training import causal_nll

    def loss(update):
        apply_adapter_tensors(session.model, update)
        return causal_nll(session.model, session.tokenizer, rows, spec.model,
                          spec.training.max_length, 4)["bits_per_token"]

    points = []
    try:
        base = loss({n: torch.zeros_like(t) if ".lora_B." in n else t for n, t in tensors.items()})
        raw_gain = base - loss(tensors)
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
                         "heldout_bits_saved_per_token": base - loss(decoded)}
                points.append(point)
                print(spec.run_id, json.dumps(point), flush=True)
        reference = max([raw_gain] + [p["heldout_bits_saved_per_token"] for p in points])
        return {"independent_rows": len(rows), "independent_raw_gain": raw_gain,
                "independent_r_star": r_star(points, reference=reference)["r_star"],
                "independent_r_star_raw": r_star(points, reference=raw_gain)["r_star"],
                "independent_points": json.dumps(points),
                "independent_example_ids": json.dumps([r.example_id for r in rows])}
    finally:
        apply_adapter_tensors(session.model, tensors)


def main() -> None:
    cells, adapters, runs, prepared, out, shard, shards = sys.argv[1:8]
    shard, shards = int(shard), int(shards)
    mode = sys.argv[8] if len(sys.argv) > 8 else "functional"
    if mode not in {"functional", "gain", "attenuation", "independent_rate"}:
        raise ValueError(f"unknown profile: {mode}")
    row_count = int(sys.argv[9]) if len(sys.argv) > 9 else (64 if mode in {"attenuation", "independent_rate"} else ROWS)
    if row_count < 1:
        raise ValueError("rows must be positive")
    split = sys.argv[10] if len(sys.argv) > 10 else "train"
    if split not in {"train", "calibration"}:
        raise ValueError(f"unknown split: {split}")
    table = sorted(csv.DictReader(open(cells)), key=lambda r: (r["model_key"], r["run_id"]))
    # Runs, not receivers, are dealt out, so one large receiver spreads over
    # every worker. A receiver's runs can differ in adapter rank, so the session
    # is re-attached whenever the adapter spec changes.
    mine = table[shard::shards]
    specs = [RunSpec.from_dict(json.loads((Path(runs) / r["run_id"] / "config.json").read_text()))
             for r in mine]
    results, session, attached = [], None, None
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
            results.append({"run_id": row["run_id"], **profile})
            print(row["run_id"], json.dumps(profile), flush=True)
            # Each completed adapter survives a later timeout or load failure.
            with open(f"{out}_{shard}.csv", "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(results[0]))
                writer.writeheader()
                writer.writerows(results)
    finally:
        if session is not None:
            session.unload()


if __name__ == "__main__":
    main()
