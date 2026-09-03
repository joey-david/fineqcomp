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
GenerationProfile = Literal["greedy", "qwen3_thinking", "deepseek_r1"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    revision: str
    backbone: Backbone
    chat: bool = False
    disable_thinking: bool = False
    generation_profile: GenerationProfile = "greedy"


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
    # Hard cap on optimizer updates, whatever the epoch count implies. Two
    # things need it: an arm comparison that varies the training rows, which
    # has to hold the update budget fixed and cannot reach one budget through
    # epochs alone when the row counts differ by orders of magnitude; and the
    # capped probes of the bit-budget study, where a run of one update over a
    # large batch is the correction the corpus asks of the frozen model. The
    # cosine schedule is built over the capped total, so a capped run still
    # ends on a decayed learning rate.
    max_updates: int | None = None
    # Whether to end on the best scored state or on the last one. Turning it
    # off separates logging from selection: a study can watch the loss curve
    # without the raw adapter becoming a maximum over checkpoints scored on
    # the same rows that later define R*.
    restore_best: bool = True
    # Which part of the response carries the loss: the whole thing, the working
    # that leads to the answer, or the answer alone. Anything but "all" needs
    # the dataset to name its answer marker.
    label_span: str = "all"


@dataclass(frozen=True)
class CodecSpec:
    key: str
    method: CodecMethod
    bits: int | None = None
    quantizer: Quantizer = "midrise"
    # Fraction of rows written one bit wider, for rates between the rungs.
    # At bits = 0 it is the fraction kept at one bit, for rates below one.
    blend: float = 0.0
    # Held-out bits saved costs a forward pass over a few hundred rows; the
    # task score costs a full test pass. Rungs that only have to place the
    # crossing turn the task score off and keep the cheap axis.
    score_task: bool = True
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
    profiles = {"greedy", "qwen3_thinking", "deepseek_r1"}
    for key, model in raw["models"].items():
        profile = str(model.get("generation_profile", "greedy"))
        if profile not in profiles:
            raise ValueError(f"model {key}: unknown generation profile {profile!r}")
    for key, codec in codecs.items():
        if codec.get("method") == "uniform":
            bits = int(codec.get("bits", 0))
            if bits not in {0, 1, 2, 3, 4, 8, 16}:
                raise ValueError(f"codec {key}: unsupported uniform bit width")
            quantizer = codec.get("quantizer", "midrise")
            if quantizer not in {"midrise", "midtread"}:
                raise ValueError(f"codec {key}: unknown quantizer {quantizer!r}")
            if bits == 16 and quantizer != "midrise":
                raise ValueError(f"codec {key}: fp16 has no quantizer choice")
            blend = float(codec.get("blend", 0.0))
            if not 0.0 <= blend < 1.0:
                raise ValueError(f"codec {key}: blend must be in [0, 1)")
            if blend and bits not in {0, 1, 2, 3}:
                raise ValueError(f"codec {key}: cannot blend {bits} bits upward")
            if bits == 0 and not blend:
                raise ValueError(f"codec {key}: a zero-bit code needs a blend")
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
    for key, training in raw["training"].items():
        span = str(training.get("label_span", "all"))
        if span not in {"all", "reasoning", "answer"}:
            raise ValueError(f"training {key}: unknown label span {span!r}")
        if span == "all":
            continue
        # A split span is meaningless without the string that starts the
        # answer, and a study that trains on one must not reach the GPU
        # before that is checked.
        users = [
            study
            for study, spec in raw["studies"].items()
            if spec.get("training") == key
        ]
        for study in users:
            for dataset_key in raw["studies"][study]["datasets"]:
                if not raw["datasets"][str(dataset_key)].get("answer_marker"):
                    raise ValueError(
                        f"dataset {dataset_key}: the {span} span needs an"
                        " answer_marker"
                    )
    for key, dataset in raw["datasets"].items():
        generation_limit = dataset.get("evaluation_max_new_tokens")
        if generation_limit is not None and int(generation_limit) < 1:
            raise ValueError(
                f"dataset {key}: evaluation_max_new_tokens must be positive"
            )
        control = dataset.get("rationale_control")
        if control is None:
            continue
        if control != "permuted":
            raise ValueError(f"dataset {key}: unknown rationale control {control!r}")
        if not dataset.get("answer_marker"):
            raise ValueError(f"dataset {key}: a rationale control needs an answer_marker")
    return raw
