from __future__ import annotations

import torch

from fineqcomp.adapters import (
    adapter_tensors,
    apply_adapter_tensors,
    attach_adapter,
    effective_adapter_rank,
    resolve_target_modules,
    seeded_a_tensor,
    unload_adapter,
)
from fineqcomp.config import AdapterSpec


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block(), Block()])


class DummyLora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(4, 2, bias=False)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 4, bias=False)}
        )


def test_seeded_a_is_reproducible_but_name_and_seed_specific():
    first = seeded_a_tensor((4, 8), 11, "x")
    assert torch.equal(first, seeded_a_tensor((4, 8), 11, "x"))
    assert not torch.equal(first, seeded_a_tensor((4, 8), 22, "x"))
    assert not torch.equal(first, seeded_a_tensor((4, 8), 11, "y"))


def test_target_resolution_honors_layer_and_projection_scope():
    spec = AdapterSpec(
        key="test",
        method="seeded_b",
        rank=4,
        target_modules=("q_proj", "v_proj"),
        last_n_layers=1,
        alpha=8,
    )
    assert resolve_target_modules(DummyModel(), spec) == [
        "layers.2.q_proj",
        "layers.2.v_proj",
    ]


def test_matched_placement_rank_tracks_reference_parameter_budget():
    model = DummyModel()
    spec = AdapterSpec(
        key="attention",
        method="full_lora",
        rank=4,
        target_modules=("q_proj",),
        last_n_layers=None,
        alpha=None,
        reference_target_modules=("q_proj", "k_proj", "v_proj"),
        reference_rank=4,
    )

    assert effective_adapter_rank(model, spec) == 12


def test_channel_tensor_selection_and_restore():
    model = DummyLora()
    seeded = adapter_tensors(model, "seeded_b")
    full = adapter_tensors(model, "full_lora")
    assert all("lora_B" in name for name in seeded)
    assert any("lora_A" in name for name in full)

    replacement = {name: torch.full_like(value, 3.0) for name, value in seeded.items()}
    apply_adapter_tensors(model, replacement)
    assert all(
        torch.all(value == 3) for value in adapter_tensors(model, "seeded_b").values()
    )


def test_peft_seeded_b_channel_has_only_trainable_b_tensors():
    from transformers import LlamaConfig, LlamaForCausalLM

    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    spec = AdapterSpec(
        key="test",
        method="seeded_b",
        rank=4,
        target_modules=("q_proj", "v_proj"),
        last_n_layers=1,
        alpha=8,
    )
    model = attach_adapter(base, spec, seed=11)
    channel = adapter_tensors(model, "seeded_b")

    assert len(channel) == 2
    assert sum(value.numel() for value in channel.values()) == 128
    assert (
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        == 128
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_A" in name
    )
    assert type(unload_adapter(model)).__name__ == "LlamaForCausalLM"
