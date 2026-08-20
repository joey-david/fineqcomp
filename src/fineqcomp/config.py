"""Campaign configuration and immutable run contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


Backbone = Literal["nf4", "bf16"]
AdapterMethod = Literal["seeded_b", "full_lora"]
CodecMethod = Literal["uniform", "loraquant"]
Quantizer = Literal["midrise", "midtread"]
RunKind = Literal["natural"]


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
    alpha: int | None
    dropout: float = 0.0
    reference_target_modules: tuple[str, ...] = ()
    reference_rank: int | None = None


@dataclass(frozen=True)
class TrainingSpec:
    epochs: int
    learning_rate: float
    effective_batch_size: int
    micro_batch_size: int
    max_length: int
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    max_grad_norm: float = 1.0
    # Score the held-out split every N optimizer updates as well as at each
    # epoch end, so best-state restore has fine-grained candidates.
    eval_every_updates: int | None = None


@dataclass(frozen=True)
class CodecSpec:
    key: str
    method: CodecMethod
    bits: int | None = None
    quantizer: Quantizer = "midrise"
    high_bits: int | None = None
    low_bits: int | None = None
    variance_ratio: float | None = None
    group_size: int = 128
    optimize_steps: int = 0


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    study: str
    kind: RunKind
    model: ModelSpec
    adapter: AdapterSpec
    seed: int
    codecs: tuple[CodecSpec, ...]
    training: TrainingSpec
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
                    "reference_target_modules": tuple(
                        raw["adapter"].get("reference_target_modules", ())
                    ),
                }
            ),
            seed=int(raw["seed"]),
            codecs=tuple(CodecSpec(**codec) for codec in raw["codecs"]),
            training=TrainingSpec(**raw["training"]),
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
    codecs = raw.get("codecs")
    if not isinstance(codecs, dict) or not codecs:
        raise ValueError("codecs must be a non-empty mapping")
    for key, codec in codecs.items():
        if codec.get("method") == "uniform":
            bits = int(codec.get("bits", 0))
            if bits not in {1, 2, 3, 4, 8, 16}:
                raise ValueError(f"codec {key}: unsupported uniform bit width")
            quantizer = codec.get("quantizer", "midrise")
            if quantizer not in {"midrise", "midtread"}:
                raise ValueError(f"codec {key}: unknown quantizer {quantizer!r}")
            if bits == 16 and quantizer != "midrise":
                raise ValueError(f"codec {key}: fp16 has no quantizer choice")
        elif codec.get("method") == "loraquant":
            if int(codec.get("high_bits", 0)) not in {2, 3}:
                raise ValueError(f"codec {key}: high_bits must be 2 or 3")
            if int(codec.get("low_bits", 1)) != 1:
                raise ValueError(f"codec {key}: low_bits must be 1")
            ratio = float(codec.get("variance_ratio", 0.0))
            if not 0.0 < ratio <= 1.0:
                raise ValueError(f"codec {key}: invalid variance_ratio")
        else:
            raise ValueError(f"codec {key}: unknown method")
    return raw
