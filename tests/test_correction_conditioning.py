import torch
import pytest

from fineqcomp.correction_conditioning import candidates, score_rows, strict_answer
from fineqcomp.data import Example


def test_direct_answer_changes_only_the_instruction():
    from fineqcomp.correction_conditioning import generation_rows
    prompt = "Solve the problem. Show concise work and finish with '#### ' followed by the answer.\n\nQuestion: 2 + 3?\nAnswer:"
    rows = [Example('x', prompt, '5', {'template': 'x'})]
    direct = generation_rows(rows, 'direct')[0]
    assert direct.prompt.split('\n\n', 1)[1] == prompt.split('\n\n', 1)[1] + ' #### '
    assert direct.response == rows[0].response and direct.metadata == rows[0].metadata
    assert 'Give only the final answer' in direct.prompt
    assert generation_rows(rows, 'cot') is rows
    with pytest.raises(ValueError, match='standard GSM prompt'):
        generation_rows([Example('x', 'unexpected prompt', '5', {})], 'direct')


def test_precision_sweep_does_not_change_the_uncompressed_tensors():
    raw = {'layer.lora_A.weight': torch.randn(8, 12), 'layer.lora_B.weight': torch.randn(10, 8)}
    cells = {key: (tensors, bits) for key, tensors, bits, _ in candidates(raw, 8, 11, {'uniform_bits': [2, 1]})}
    for bits in (2, 1):
        tensors, precision = cells[f'uniform_b{bits}']
        assert tensors is raw and precision == bits


def test_compression_analysis_requires_each_configured_state():
    from fineqcomp.cot_compression_analysis import (
        answer_only, expected_states, paired_strict_difference,
    )
    config = dict(uniform_bits=[2, 1], scales=[], svd_ranks=[1, 4],
                  svd_binary=True, random_draws=0)
    assert expected_states(config, 16) == {
        'base', 'raw', 'uniform_b2', 'uniform_b1', 'mask_half',
        'svd_r1', 'svd_r4', 'svd_r2_binary',
    }
    assert answer_only('#### -1,204.5\n')
    assert not answer_only('#### 13 * 8 = 104\n')
    assert not answer_only('#### 70 sushi rolls\n')
    left = [dict(example_id='a', template='x', strict_correct=True),
            dict(example_id='b', template='y', strict_correct=False)]
    right = [dict(example_id='b', template='y', strict_correct=True),
             dict(example_id='a', template='x', strict_correct=True)]
    paired = paired_strict_difference(left, right)
    assert paired['difference'] == -.5
    assert (paired['both_correct'], paired['left_only'], paired['right_only']) == (1, 0, 1)


def test_scalar_control_changes_update_by_scale_not_scale_squared():
    raw = {'layer.lora_A.weight': torch.randn(8, 12), 'layer.lora_B.weight': torch.randn(10, 8)}
    controls = {key: tensors for key, tensors, _, _ in candidates(raw, 8, 11)}
    actual = controls['scale_0.5']['layer.lora_B.weight'] @ controls['scale_0.5']['layer.lora_A.weight']
    assert torch.allclose(actual, .5 * raw['layer.lora_B.weight'] @ raw['layer.lora_A.weight'])


def test_unfinished_arithmetic_is_not_a_strict_answer():
    assert strict_answer('I calculate 13 + 29 = 42.') is None
    assert strict_answer('The answer is: 42') == '42'
    assert strict_answer(r'Final result: \boxed{42}') == '42'
    assert strict_answer('In total, 1084 bananas. #### 1084') == '1084'


def test_prefix_crossover_keeps_early_answers_visible():
    row = Example('x', 'Question:', '42', {'template': 'x', 'instance': 5, 'split': 'test'})
    record = {'response': '', 'hit_generation_limit': False}
    prefix = {'response': 'The answer is: 42', 'terminated_with_eos': True}
    actual = score_rows([row], [record], [prefix])[0]
    assert actual['strict_correct'] and actual['prefix_terminated']
