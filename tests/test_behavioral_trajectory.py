import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import read_json
from fineqcomp.behavioral_trajectory import (
    codec_candidates, evaluate_checkpoint, matched_field, paired_interval, report, score,
    select_files, train_cell,
)
from fineqcomp.config import AdapterSpec, ModelSpec, RunSpec, TrainingSpec
from fineqcomp.data import Example
from fineqcomp.modeling import ModelSession
from fineqcomp.training import train_adapter


@pytest.fixture
def tiny():
    torch.manual_seed(17)
    vocab = {'[PAD]': 0, '[EOS]': 1, '[UNK]': 2,
             'Question': 3, 'Answer': 4, '1': 5, '2': 6, '3': 7}
    backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
        pad_token='[PAD]', eos_token='[EOS]', unk_token='[UNK]')
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1, bos_token_id=None))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    spec = ModelSpec('tiny', 'tiny', 'local', 'bf16')
    adapter = AdapterSpec('tiny', 'full_lora', 4, ('q_proj', 'v_proj'), None, 8)
    run = RunSpec('tiny', 'trajectory', 'natural', spec, adapter, 11, (),
        TrainingSpec(epochs=2, learning_rate=0.01, effective_batch_size=2,
            micro_batch_size=1, max_length=32, warmup_ratio=0,
            max_updates=4, restore_best=False), 'tiny')
    session = ModelSession(spec, model, tokenizer)
    session.attach(adapter, 11)
    examples = [Example(str(i), f'Question {i} Answer', ' 2', {'evaluator': 'gsm8k'})
                for i in range(8)]
    config = {'checkpoints': [1, 2, 4], 'updates': 4, 'intervention_step': 2,
        'intervention_rank': 1, 'interventions': ['restart', 'truncate', 'random'],
        'ranks': [1, 4], 'bits': [1, 2], 'generation_batch_size': 2,
        'max_new_tokens': 2, 'retentions': [0.9], 'minimum_reference_change': 0.0,
        'bootstrap_draws': 20, 'analysis_seed': 11}
    data = {'aligned': examples, 'permuted': examples, 'calibration': examples[:2],
            'gsm8k': examples[2:4], 'symbolic': examples[4:6]}
    return session, run, config, data


def test_callback_does_not_change_schedule_or_weights(tiny, tmp_path):
    session, run, _, data = tiny
    initial = adapter_tensors(session.model, 'full_lora')
    train_adapter(session.model, session.tokenizer, data['aligned'], data['calibration'],
        run.model, run.training, 11, tmp_path / 'plain.jsonl')
    expected = adapter_tensors(session.model, 'full_lora')
    apply_adapter_tensors(session.model, initial)
    rates, snapshots = [], []

    def save(step, optimizer):
        rates.append(optimizer.param_groups[0]['lr'])
        snapshots.append(adapter_tensors(session.model, 'full_lora'))

    train_adapter(session.model, session.tokenizer, data['aligned'], data['calibration'],
        run.model, run.training, 11, tmp_path / 'saved.jsonl', on_update=save)
    assert len(snapshots) == 4
    assert rates[-1] == 0
    assert all(rates[i] > rates[i + 1] for i in range(3))
    for name, value in adapter_tensors(session.model, 'full_lora').items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['truncate', 'random'])
def test_intervention_keeps_dead_directions_zero(tiny, tmp_path, kind):
    session, run, config, data = tiny
    train_cell(session, run, {'kind': kind, 'arm': 'aligned'}, data, config, tmp_path)
    before = torch.load(tmp_path / 'before_intervention.pt', weights_only=True)
    final = torch.load(tmp_path / 'step4.pt', weights_only=True)
    assert any(t.any() for n, t in before.items() if '.lora_B.' in n)
    for name, tensor in final.items():
        tail = tensor[1:] if '.lora_A.' in name else tensor[:, 1:]
        assert torch.count_nonzero(tail) == 0


def test_fixed_threshold_can_be_unreachable_and_preserves_damage_direction():
    rows = [{'key': 'small', 'accuracy': 0.3, 'file_bits': 100},
            {'key': 'large', 'accuracy': 0.5, 'file_bits': 200}]
    assert select_files(rows, 0.1, 0.8, [0.9], 0.05)[0]['status'] == 'unreachable'
    selected = select_files(rows, 0.8, 0.2, [0.75], 0.05)[0]
    assert selected['key'] == 'small'
    assert selected['direction'] == 'damage'
    assert select_files(rows, 0.8, 0.81, [0.9], 0.05)[0]['status'] == 'reference_too_small'


def test_recovery_controls_partition_the_update_and_match_norm():
    torch.manual_seed(31)
    tensors = {'layer.lora_A.default.weight': torch.randn(4, 10),
               'layer.lora_B.default.weight': torch.randn(12, 4)}
    candidates = {key: value for key, value, _ in codec_candidates(tensors,
        {'ranks': [1, 4], 'bits': [1, 16], 'spectral_controls': True}, 11)}

    def product(value):
        return value['layer.lora_B.default.weight'] @ value['layer.lora_A.default.weight']

    full = product(tensors)
    head = product(candidates['r1_b16'])
    torch.testing.assert_close(head + product(candidates['tail_r3_b16']), full)
    torch.testing.assert_close(product(candidates['norm_full_r4_b16']).norm(), head.norm())
    assert torch.linalg.matrix_rank(product(candidates['random_r1_b16'])) == 1


def test_binary_norm_controls_and_linear_scale(tmp_path):
    from fineqcomp.codec import encode_tensor_map, decode_adapter_tensor_map
    torch.manual_seed(45)
    tensors = {'layer.lora_A.default.weight': torch.randn(4, 10),
               'layer.lora_B.default.weight': torch.randn(12, 4)}
    candidates = {key: value for key, value, _ in codec_candidates(tensors,
        {'ranks': [1], 'bits': [1], 'scale_controls': True, 'scale_sweep': [0.1, 1.0]}, 11)}

    def product(value):
        return value['layer.lora_B.default.weight'] @ value['layer.lora_A.default.weight']

    path = tmp_path / 'binary.fqcb'
    encode_tensor_map(candidates['r1_b1'], path, 1)
    _, binary = decode_adapter_tensor_map(path)
    for label in ('head', 'full'):
        torch.testing.assert_close(product(candidates[f'binary_norm_{label}_b16']).norm(),
                                   product(binary).norm())
        torch.testing.assert_close(product(candidates[f'scale_{label}_0.1_b16']),
                                   product(candidates[f'scale_{label}_1_b16']) * 0.1)
    torch.testing.assert_close(product(candidates['scale_full_1_b16']), product(tensors))


def test_transformer_train_decode_generate_select_test_and_report(tiny, tmp_path):
    session, run, config, data = tiny
    config['test_all_candidates'] = True
    config['spectral_controls'] = True
    config['bits'] = [1, 16]
    root = tmp_path / 'tiny/aligned/seed11/trajectory'
    cell = {'kind': 'trajectory', 'arm': 'aligned', 'slug': 'tiny/aligned/seed11/trajectory'}
    baseline = {part: score(session, run, data[part], config, root / f'base_{part}.json')
                for part in ('calibration', 'gsm8k', 'symbolic')}
    train_cell(session, run, cell, data, config, root)
    final = adapter_tensors(session.model, 'full_lora')
    reference = {part: score(session, run, data[part], config, root / f'reference_{part}.json')
                 for part in ('calibration', 'gsm8k', 'symbolic')}
    # Equal reference and baseline force a reachable software-only threshold.
    reference['calibration'] = baseline['calibration']
    evaluate_checkpoint(session, run, final, reference, baseline, data, config, root / 'step4')
    selected = read_json(root / 'step4/selection.json')
    assert selected[0]['status'] == 'selected'
    path = root / 'step4' / f'{selected[0]["key"]}.fqcb'
    assert path.stat().st_size * 8 == selected[0]['file_bits']
    # Exercise resuming after the lock was written but a test output was lost.
    test = root / 'step4' / f'{selected[0]["key"]}_gsm8k.json'
    test.unlink()
    evaluate_checkpoint(session, run, final, reference, baseline, data, config, root / 'step4')
    assert test.exists()
    from fineqcomp.artifacts import write_json
    write_json(root / 'complete.json', {'complete': True})
    result = report({'config': config, 'cells': [cell]}, tmp_path)
    assert result['status'] == 'complete'
    assert len(result['rows']) == 2
    assert len(result['candidate_results']) == 14
    assert paired_interval(test.with_suffix('.jsonl'), test.with_suffix('.jsonl'), 20, 11) == [0, 0]


def test_matched_field_runs_exact_exposures_from_one_prefix(tiny, tmp_path, monkeypatch):
    from fineqcomp import config as config_module, program_selection as ps
    session, run, config, _ = tiny
    session.unload()
    ps_config = ps.load_config(__import__('pathlib').Path('configs/program_selection.yaml'))
    ps_config.update(diagnostic_rows=8, targeted_candidates=8, probe_rows=8,
                     ambiguous_eval_rows=8, generation_max_new_tokens=2)
    monkeypatch.setattr(ps, 'load_config', lambda _: ps_config)
    monkeypatch.setattr(ModelSession, 'load', lambda _: session)
    monkeypatch.setattr(config_module, 'load_campaign', lambda _: {
        'models': {'tiny': {'name': 'tiny', 'revision': 'local', 'backbone': 'bf16'}},
        'adapters': {'all_linear_r16': {'method': 'full_lora', 'rank': 4,
            'target_modules': ['q_proj', 'v_proj'], 'last_n_layers': None,
            'alpha': 8, 'dropout': 0}}})
    config.update(field_models=['tiny'], seeds=[11], field_config='unused',
                  field_campaign='unused', field_target='position_2',
                  field_incumbent='named_code', field_exposures=[1, 8])
    matched_field(config, tmp_path, 0)
    root = tmp_path / 'field/tiny/seed11'
    assert (root / 'complete.json').exists()
    results = [read_json(p) for p in root.glob('*/result.json')]
    assert sorted(r['exposures'] for r in results) == [0, 1, 1, 8, 8]
    assert all(r['training']['train_examples'] == 512 for r in results)
    assert all(r['training']['optimizer_updates'] == 32 for r in results)
