"""Campaign configuration and immutable run contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


Backbone = Literal["nf4", "bf16"]
AdapterMethod = Literal["seeded_b", "full_lora"]
RunKind = Literal["synthetic", "controlled", "natural"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    revision: str
    backbone: Backbone
    chat: bool = False
    disable_thinking: bool = False


@dataclass(frozen=True)
class AdapterSpec:
    key: str
    method: AdapterMethod
    rank: int
    target_modules: tuple[str, ...]
    last_n_layers: int | None
    alpha: int
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainingSpec:
    epochs: int
    learning_rate: float
    effective_batch_size: int
    micro_batch_size: int
    max_length: int
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    study: str
    kind: RunKind
    model: ModelSpec
    adapter: AdapterSpec
    seed: int
    precisions: tuple[int, ...]
    clip_percentiles: tuple[float, ...]
    training: TrainingSpec
    family_count: int | None = None
    binding_count: int | None = None
    dataset_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunSpec":
        return cls(
            run_id=str(raw["run_id"]),
            study=str(raw["study"]),
            kind=str(raw["kind"]),  # type: ignore[arg-type]
            model=ModelSpec(**raw["model"]),
            adapter=AdapterSpec(
                **{
                    **raw["adapter"],
                    "target_modules": tuple(raw["adapter"]["target_modules"]),
                }
            ),
            seed=int(raw["seed"]),
            precisions=tuple(map(int, raw["precisions"])),
            clip_percentiles=tuple(map(float, raw["clip_percentiles"])),
            training=TrainingSpec(**raw["training"]),
            family_count=(
                int(raw["family_count"])
                if raw.get("family_count") is not None
                else None
            ),
            binding_count=(
                int(raw["binding_count"])
                if raw.get("binding_count") is not None
                else None
            ),
            dataset_key=raw.get("dataset_key"),
        )


def load_campaign(path: str | Path) -> dict[str, Any]:
    """Load and check a campaign YAML file."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError(f"{source}: expected campaign version 1")
    for key in ("models", "adapters", "training", "datasets", "studies"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"{source}: missing mapping {key!r}")
    precisions = raw.get("precisions")
    if not precisions or any(int(bits) not in {2, 3, 4, 8, 16} for bits in precisions):
        raise ValueError("precisions must use 2, 3, 4, 8, or 16 bits")
    return raw
