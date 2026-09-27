

def test_attenuation_profile_scales_effective_update_and_restores_weights(monkeypatch):
    from types import SimpleNamespace
    import pytest
    import torch
    from fineqcomp.studies.generalisation import attenuation_profile

    raw = {"m.lora_A.default.weight": torch.tensor([[1., -2., 3.]]),
           "m.lora_B.default.weight": torch.tensor([[2.], [-1.]])}
    current, seen = {}, []

    def apply(model, tensors):
        current.clear()
        current.update({n: t.clone() for n, t in tensors.items()})

    def loss(*args):
        a, b = current["m.lora_A.default.weight"], current["m.lora_B.default.weight"]
        alpha = float(b[0, 0] / raw["m.lora_B.default.weight"][0, 0])
        assert torch.equal(a, raw["m.lora_A.default.weight"])
        seen.append(alpha)
        return {"bits_per_token": 2 - alpha, "nll_tokens": 3}

    monkeypatch.setattr("fineqcomp.studies.generalisation.apply_adapter_tensors", apply)
    monkeypatch.setattr("fineqcomp.training.causal_nll", loss)
    profile = attenuation_profile(SimpleNamespace(model=None, tokenizer=None), raw, [None], None, 8)
    assert seen == [0, 0.125, 0.25, 0.5, 1]
    # For a rank-one update, <Q(BA), BA>/||BA||^2 is kappa(A)*kappa(B).
    expected = (36 / (3 * 14)) * (9 / (2 * 5))
    assert profile["attenuation_binary_projection"] == pytest.approx(expected)
    assert profile["attenuation_binary_residual"] == pytest.approx(expected - expected ** 2)
    assert all(torch.equal(current[n], t) for n, t in raw.items())

    def fail(*args):
        raise RuntimeError("probe failed")

    monkeypatch.setattr("fineqcomp.training.causal_nll", fail)
    with pytest.raises(RuntimeError, match="probe failed"):
        attenuation_profile(SimpleNamespace(model=None, tokenizer=None), raw, [None], None, 8)
    assert all(torch.equal(current[n], t) for n, t in raw.items())


def test_spectral_profile_reads_rank_and_pair_structure():
    import torch
    from fineqcomp.studies.generalisation import spectral_profile
    torch.manual_seed(0)
    rank1 = {"m.lora_A.default.weight": torch.randn(4, 12), "m.lora_B.default.weight": torch.zeros(10, 4)}
    rank1["m.lora_B.default.weight"][:, 0] = torch.randn(10)
    p = spectral_profile(rank1)
    assert abs(p["spectrum_top1"] - 1.0) < 1e-9 and abs(p["spectrum_effective_rank"] - 1.0) < 1e-6
    q, _ = torch.linalg.qr(torch.randn(12, 4))
    u, _ = torch.linalg.qr(torch.randn(10, 4))
    orthogonal = {"m.lora_A.default.weight": q.T, "m.lora_B.default.weight": u}
    p = spectral_profile(orthogonal)
    assert abs(p["spectrum_pair_overlap"] - 1.0) < 1e-9 and abs(p["spectrum_effective_rank"] - 4.0) < 1e-6


def test_functional_profile_joins_adapter_base_and_data():
    import torch
    from types import SimpleNamespace
    from peft import LoraConfig, get_peft_model
    from fineqcomp.data import Example
    from fineqcomp.studies.generalisation import functional_profile
    from test_relative_info import _tiny_causal_session
    session = _tiny_causal_session()
    session.model = get_peft_model(session.model, LoraConfig(r=4, lora_alpha=8, target_modules=["v_proj"], layers_to_transform=[3]))
    torch.manual_seed(1)
    for name, p in session.model.named_parameters():
        if "lora_B" in name:
            p.data = torch.randn_like(p) * 0.1
    rows = [Example(f"e{i}", f"question {i}", f" answer {i} is here", {}) for i in range(4)]
    spec = SimpleNamespace(chat=False)
    first = functional_profile(session, rows, spec, 64)
    assert 1.0 <= first["functional_effective_rank"] <= 4.0
    assert 0.0 < first["functional_top1"] <= 1.0
    for name, p in session.model.named_parameters():
        if "lora_B" in name:
            p.data *= 3.0
    scaled = functional_profile(session, rows, spec, 64)
    # One site, last layer: nothing upstream moves, so tripling B multiplies the
    # update's output energy by exactly 9 and leaves its shape alone.
    assert abs(scaled["functional_log_relative_energy"] - first["functional_log_relative_energy"] - 2 * torch.log(torch.tensor(3.0)).item()) < 1e-6
    assert abs(scaled["functional_top1"] - first["functional_top1"]) < 1e-6


def test_gain_profile_scores_each_direction_on_the_data():
    import torch
    from types import SimpleNamespace
    from peft import LoraConfig, get_peft_model
    from fineqcomp.adapters import adapter_tensors
    from fineqcomp.data import Example
    from fineqcomp.studies.generalisation import gain_profile
    from test_relative_info import _tiny_causal_session
    session = _tiny_causal_session()
    session.model = get_peft_model(session.model, LoraConfig(r=4, lora_alpha=8, target_modules=["v_proj"]))
    torch.manual_seed(2)
    tensors = {n: (torch.randn_like(t) * 0.3) for n, t in adapter_tensors(session.model, "full_lora").items()}
    rows = [Example(f"e{i}", f"question {i}", f" answer {i} is here", {}) for i in range(4)]
    profile = gain_profile(session, tensors, rows, SimpleNamespace(chat=False), 64, 2)
    assert 1.0 <= profile["gain_directions_90"] <= 4.0
    assert 0.0 <= profile["gain_negative_share"] <= 1.0
    assert all(value == value for value in profile.values())
