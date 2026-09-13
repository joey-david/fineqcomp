from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from fineqcomp.policy_complexity import cell_data, contract, load_config, prepare
from fineqcomp.policy_complexity_analysis import budget, learning_area, reduce

CONFIG = Path(__file__).parents[1] / 'configs/policy_complexity/smoke.json'


def test_lock_rejects_changed_contract(tmp_path):
    config = load_config(CONFIG)
    locked = prepare(config, tmp_path)
    assert locked == prepare(config, tmp_path)
    config['seeds'] = [22]
    with pytest.raises(ValueError, match='differs'):
        prepare(config, tmp_path)


def test_all_planned_cells_obey_data_contract():
    config = load_config(CONFIG.with_name('followups.json'))
    for cell in contract(config)['cells']:
        data = cell_data(cell, config)
        assert len({r.metadata['entity_key'] for r in data['train']}) == cell['n']
        assert len(data['train']) == cell['n'] * cell['repeat']


def test_area_distinguishes_fast_rule_from_slow_rule_and_keeps_failures():
    easy = learning_area([16, 64, 256], [.1, .1, .1], 1., .1, .1)
    hard = learning_area([16, 64, 256], [.9, .8, .1], 1., .1, .1)
    assert easy['area'] < hard['area']
    assert learning_area([16], [1.1], 1., 1.1, 1.2)['area'] is None
    assert learning_area([16], [.1], 1., .1, .5)['status'] == 'optimization_unresolved'
    assert learning_area([16, 32], [-1., 0.], 1., 0., 0.)['residual'][0] == -1.


def test_budget_never_selects_using_verification():
    scores = lambda s, v: {'selection_recall': {'nll': s}, 'verification_recall': {'nll': v}}
    record = dict(base_parameters=100, base=scores(1., 1.), grid=[
        dict(key='small', file_bits=100, scores=scores(.1, .4)),
        dict(key='big', file_bits=200, scores=scores(.2, .1))])
    result = budget(record, .25)
    assert result['key'] == 'small' and result['status'] == 'verification_failed'
    assert budget(record, 2.)['file_bits'] == 0
    assert budget(record, .01)['status'] == 'target_unattained'


def test_missing_cells_are_not_silently_removed(tmp_path):
    prepare(load_config(CONFIG), tmp_path)
    result = reduce(tmp_path)
    assert not result['complete'] and result['completed_cells'] == 0
    assert len(result['missing_cells']) == result['expected_cells']


def test_simultaneous_array_preparation_has_one_immutable_lock(tmp_path):
    config = load_config(CONFIG)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda _: prepare(config, tmp_path), range(8)))
    assert all(record == records[0] for record in records)


def test_real_tiny_model_runs_training_and_decoded_codec(tmp_path, monkeypatch):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from fineqcomp.artifacts import read_json
    from fineqcomp.modeling import ModelSession
    from fineqcomp.policy_complexity import run_base, run_cell

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        def encode(self, text, add_special_tokens=False):
            return ([1] if add_special_tokens else []) + [3 + ord(c) % 120 for c in text]

    def load(spec):
        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=256, use_cache=False))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return ModelSession(spec, model, Tokenizer())

    monkeypatch.setattr(ModelSession, 'load', load)
    config = load_config(CONFIG)
    config['models'] = [dict(key='tiny', name='unused', revision='test', backbone='bf16')]
    config['codec'] = dict(ranks=[1], bits=[1])
    locked = prepare(config, tmp_path)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        run_base(config, tmp_path, 0)
        run_cell(config, tmp_path, locked['cells'][0], 0)
    finally:
        torch.set_num_threads(previous_threads)
    record = read_json(tmp_path / 'cells/0000/complete.json')
    assert record['training']['optimizer_updates'] == 2
    assert [r['step'] for r in record['checkpoints']] == [1, 2]
    assert record['grid'][0]['file_bits'] == 8 * (tmp_path / 'cells/0000/codecs/r1_b1.fqcb').stat().st_size
    assert len(record['grid'][0]['scores']['verification_recall']['rows']) == 16
    assert reduce(tmp_path)['complete']
