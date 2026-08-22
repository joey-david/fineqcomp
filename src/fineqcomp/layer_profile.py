"""Where in the network do the necessary adapter bits live?

Two measurements, both on adapters that already exist. Every finished run keeps
`raw_channel.pt`, so nothing here trains anything: the work is forward passes.

`allocation_sweep` codes bands of transformer layers at different rates under a
matched total budget -- starve the early third and keep the late third, then the
reverse -- and scores how much of the behavioural change survives. Experiment 1
found that allocating bits per *row* by reconstruction error never beats a
uniform code; per *layer* is a different axis, and it is the one that says where
the behaviour is written.

`representation_shift` measures how far the hidden states move at each layer,
which is an axis the rate-distortion curve never touches. R* is chosen for
output fidelity and scored on output fidelity; a representational measure is
not, so agreement between them is evidence rather than circularity.

`distribution_shift` answers a different question. Held-out bits saved counts
only the probability the model assigned to the token that actually appeared, so
everything the adapter did to the rest of the distribution is invisible to it.
That measure ranks datasets correctly inside one corpus and backwards across
corpora: Magicoder and hh-rlhf save a third of what the math arms save and need
the largest adapters in the campaign. The divergence between the base and
adapted output distributions counts the whole reshaping, on the same tokens and
in the same units, and is the candidate for what the observed-token measure is
missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from fineqcomp.adapters import apply_adapter_tensors, trainable_state
from fineqcomp.codec import decode_layer_groups, encode_layer_groups, layer_groups
from fineqcomp.data import Example
from fineqcomp.modeling import (
    CausalExampleDataset,
    ModelSession,
    causal_collate,
    model_device,
)
from fineqcomp.training import causal_nll


# Pre-registered plans, three bands of layers, as (bits, blend) per band. Each
# block holds the mean nominal rate fixed so the comparison is at a matched
# budget and only the placement of the bits changes.
ALLOCATION_PLANS: dict[str, dict[str, tuple[int, float]]] = {
    # mean nominal rate 1.0
    "uniform_1": {"g0": (1, 0.0), "g1": (1, 0.0), "g2": (1, 0.0)},
    "early_starved_1": {"g0": (0, 0.25), "g1": (1, 0.375), "g2": (1, 0.375)},
    "middle_starved_1": {"g0": (1, 0.375), "g1": (0, 0.25), "g2": (1, 0.375)},
    "late_starved_1": {"g0": (1, 0.375), "g1": (1, 0.375), "g2": (0, 0.25)},
    # mean nominal rate 0.5
    "uniform_05": {"g0": (0, 0.5), "g1": (0, 0.5), "g2": (0, 0.5)},
    "early_starved_05": {"g0": (0, 0.125), "g1": (0, 0.6875), "g2": (0, 0.6875)},
    "middle_starved_05": {"g0": (0, 0.6875), "g1": (0, 0.125), "g2": (0, 0.6875)},
    "late_starved_05": {"g0": (0, 0.6875), "g1": (0, 0.6875), "g2": (0, 0.125)},
}


def load_raw_channel(run_dir: Path) -> dict[str, torch.Tensor]:
    """The trained adapter a finished run left behind."""
    path = Path(run_dir) / "raw_channel.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no saved adapter to profile")
    tensors = torch.load(path, map_location="cpu")
    return {name: value.float() for name, value in tensors.items()}


def weight_profile(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    """How far training moved each band, in weight space. CPU only, no model."""
    bands = layer_groups(tensors, 3)
    profile = {}
    for key, names in bands.items():
        norm = sum(float((tensors[name] ** 2).sum()) for name in names)
        values = sum(tensors[name].numel() for name in names)
        profile[key] = {
            "tensors": len(names),
            "values": values,
            "squared_norm": norm,
            "rms": (norm / max(values, 1)) ** 0.5,
        }
    return profile


@torch.no_grad()
def _pooled_states(
    session: ModelSession, rows: list[Example], max_length: int, batch_size: int
) -> torch.Tensor:
    """Mean hidden state per layer per row, pooled over real tokens.

    Pooling keeps this to a few megabytes. Storing every position for a 7B
    across 32 layers would be gigabytes, and the question here is how far a
    layer's representation moves, not where in the sequence it moved.
    """
    device = model_device(session.model)
    session.model.eval()
    collected: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = session.tokenizer(
            [f"{row.prompt}{row.response}" for row in batch],
            return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = session.model(**encoded, output_hidden_states=True)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = torch.stack(
            [
                (state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                for state in output.hidden_states
            ]
        )
        collected.append(pooled.cpu())
    return torch.cat(collected, dim=1)


def representation_shift(
    session: ModelSession,
    rows: list[Example],
    base_state: dict[str, torch.Tensor],
    adapter: dict[str, torch.Tensor],
    max_length: int = 512,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Per-layer distance between the base and adapted hidden states."""
    apply_adapter_tensors(session.model, base_state)
    before = _pooled_states(session, rows, max_length, batch_size)
    apply_adapter_tensors(session.model, adapter)
    after = _pooled_states(session, rows, max_length, batch_size)
    apply_adapter_tensors(session.model, base_state)

    layers = []
    for index in range(before.shape[0]):
        a, b = before[index], after[index]
        delta = (b - a).norm(dim=-1)
        scale = a.norm(dim=-1).clamp(min=1e-6)
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        layers.append(
            {
                "layer": index,
                "relative_shift": float((delta / scale).mean()),
                "cosine": float(cosine.mean()),
            }
        )
    return {"rows": len(rows), "layers": layers}


_LN2 = 0.6931471805599453


def _divergences(base_logits: torch.Tensor, adapted_logits: torch.Tensor,
                 observed: torch.Tensor, chunk: int = 4096) -> dict[str, float]:
    """Summed divergences over one batch's scored positions, in bits.

    Chunked over positions because a 32k vocabulary at a thousand positions is
    128 MB per distribution and this holds four of them at once.
    """
    totals = {"forward_kl": 0.0, "reverse_kl": 0.0, "jensen_shannon": 0.0,
              "total_variation": 0.0, "observed_bits_saved": 0.0}
    for start in range(0, base_logits.shape[0], chunk):
        stop = start + chunk
        log_base = torch.log_softmax(base_logits[start:stop].float(), dim=-1)
        log_adapted = torch.log_softmax(adapted_logits[start:stop].float(), dim=-1)
        base = log_base.exp()
        adapted = log_adapted.exp()
        log_mean = ((base + adapted) / 2).clamp_min(1e-12).log()
        picked = observed[start:stop].unsqueeze(-1)
        totals["forward_kl"] += float(
            (adapted * (log_adapted - log_base)).sum()
        )
        totals["reverse_kl"] += float((base * (log_base - log_adapted)).sum())
        totals["jensen_shannon"] += float(
            (adapted * (log_adapted - log_mean)).sum()
            + (base * (log_base - log_mean)).sum()
        ) / 2
        totals["total_variation"] += float((base - adapted).abs().sum()) / 2
        totals["observed_bits_saved"] += float(
            (log_adapted.gather(-1, picked) - log_base.gather(-1, picked)).sum()
        )
    for key in totals:
        if key != "total_variation":
            totals[key] /= _LN2
    return totals


@torch.no_grad()
def distribution_shift(
    session: ModelSession,
    rows: list[Example],
    base_state: dict[str, torch.Tensor],
    adapter: dict[str, torch.Tensor],
    model_spec: Any,
    max_length: int,
    batch_size: int,
    label_span: str = "all",
    answer_marker: str | None = None,
) -> dict[str, Any]:
    """How far the output distribution moved, averaged over scored tokens.

    The token set is exactly the one `causal_nll` scores -- same dataset, same
    `-100` label mask -- so the divergences and the held-out bits saved that R*
    is built on are means over the same positions and can be compared directly.
    `observed_bits_saved` recomputes that quantity here as a check on the
    alignment; it should reproduce the run's own figure.
    """
    dataset = CausalExampleDataset(
        session.tokenizer, rows, model_spec, max_length, label_span, answer_marker
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: causal_collate(batch, session.tokenizer.pad_token_id),
    )
    device = model_device(session.model)
    session.model.eval()
    totals: dict[str, float] = {}
    scored = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        # A causal model predicts position t+1 from position t, so a label at
        # index i is read off the logits at i-1.
        selected = batch["labels"][:, 1:] != -100
        if not bool(selected.any()):
            continue
        observed = batch["labels"][:, 1:][selected]
        apply_adapter_tensors(session.model, base_state)
        base_logits = session.model(**batch).logits[:, :-1, :][selected]
        apply_adapter_tensors(session.model, adapter)
        adapted_logits = session.model(**batch).logits[:, :-1, :][selected]
        batch_totals = _divergences(base_logits, adapted_logits, observed)
        for key, value in batch_totals.items():
            totals[key] = totals.get(key, 0.0) + value
        scored += int(selected.sum())
        del base_logits, adapted_logits
    apply_adapter_tensors(session.model, adapter)
    if not scored:
        raise ValueError("no scored tokens: check label_span and answer_marker")
    return {
        "rows": len(rows),
        "scored_tokens": scored,
        **{f"{key}_per_token": value / scored for key, value in totals.items()},
    }


def allocation_sweep(
    session: ModelSession,
    rows: list[Example],
    base_state: dict[str, torch.Tensor],
    adapter: dict[str, torch.Tensor],
    work_dir: Path,
    model_spec: Any,
    max_length: int,
    batch_size: int,
    plans: dict[str, dict[str, tuple[int, float]]] | None = None,
) -> dict[str, Any]:
    """Score every layer allocation on held-out bits saved."""
    plans = plans or ALLOCATION_PLANS
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    def scored() -> float:
        return float(
            causal_nll(
                session.model, session.tokenizer, rows, model_spec,
                max_length, batch_size,
            )["total_bits"]
        )

    apply_adapter_tensors(session.model, base_state)
    base_bits = scored()
    apply_adapter_tensors(session.model, adapter)
    raw_bits = scored()

    results = []
    try:
        for key in sorted(plans):
            stem = work_dir / f"plan_{key}"
            stats = encode_layer_groups(adapter, stem, plans[key])
            decoded = decode_layer_groups(stem, plans[key])
            apply_adapter_tensors(session.model, decoded)
            coded_bits = scored()
            results.append(
                {
                    "plan": key,
                    "bands": {k: list(v) for k, v in plans[key].items()},
                    "file_bits": stats["file_bits"],
                    "effective_bits_per_value": stats["effective_bits_per_value"],
                    "relative_rmse": stats["relative_rmse"],
                    "bits_saved": base_bits - coded_bits,
                    "retained_gain": (
                        (base_bits - coded_bits) / (base_bits - raw_bits)
                        if base_bits > raw_bits
                        else None
                    ),
                }
            )
            for band in plans[key]:
                stem.with_name(f"{stem.name}.{band}.fqcb").unlink(missing_ok=True)
    finally:
        apply_adapter_tensors(session.model, adapter)
    return {
        "base_total_bits": base_bits,
        "raw_total_bits": raw_bits,
        "raw_bits_saved": base_bits - raw_bits,
        "plans": results,
    }


def profile_run(
    session: ModelSession,
    run_dir: Path,
    calibration: list[Example],
    model_spec: Any,
    work_dir: Path,
    max_length: int = 1024,
    batch_size: int = 4,
    probe_rows: int = 32,
    measures: Iterable[str] = ("weights", "representation", "allocation"),
    label_span: str = "all",
    answer_marker: str | None = None,
) -> dict[str, Any]:
    """The selected measurements for one finished run.

    `measures` exists because the three cost wildly different amounts. The
    allocation sweep codes and scores eight plans; the distribution shift is two
    forward passes per batch. A campaign-wide pass over one of them should not
    pay for the others.
    """
    wanted = set(measures)
    unknown = wanted - {"weights", "representation", "allocation", "distribution"}
    if unknown:
        raise ValueError(f"unknown measures: {sorted(unknown)}")
    adapter = load_raw_channel(Path(run_dir))
    base_state = {name: torch.zeros_like(value) for name, value in adapter.items()}
    # Only lora_B is zeroed to recover the base; a full-LoRA adapter stores A
    # as well, and zeroing both is equivalent because the product is zero.
    record: dict[str, Any] = {"run_dir": str(run_dir), "measures": sorted(wanted)}
    if "weights" in wanted:
        record["weights"] = weight_profile(adapter)
    if "representation" in wanted:
        record["representation"] = representation_shift(
            session, calibration[:probe_rows], base_state, adapter,
            max_length=min(max_length, 512), batch_size=max(batch_size, 1),
        )
    if "distribution" in wanted:
        record["distribution"] = distribution_shift(
            session, calibration, base_state, adapter, model_spec,
            max_length, batch_size, label_span, answer_marker,
        )
    if "allocation" in wanted:
        record["allocation"] = allocation_sweep(
            session, calibration, base_state, adapter, work_dir,
            model_spec, max_length, batch_size,
        )
    return record
