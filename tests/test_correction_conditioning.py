import torch

from fineqcomp.correction_conditioning import candidates, score_rows, strict_answer
from fineqcomp.data import Example


def test_scalar_control_changes_update_by_scale_not_scale_squared():
    raw = {'layer.lora_A.weight': torch.randn(8, 12), 'layer.lora_B.weight': torch.randn(10, 8)}
    controls = {key: tensors for key, tensors, _, _ in candidates(raw, 8, 11)}
    actual = controls['scale_0.5']['layer.lora_B.weight'] @ controls['scale_0.5']['layer.lora_A.weight']
    assert torch.allclose(actual, .5 * raw['layer.lora_B.weight'] @ raw['layer.lora_A.weight'])


def test_unfinished_arithmetic_is_not_a_strict_answer():
    assert strict_answer('I calculate 13 + 29 = 42.') is None
    assert strict_answer('The answer is: 42') == '42'
    assert strict_answer(r'Final result: \boxed{42}') == '42'


def test_prefix_crossover_keeps_early_answers_visible():
    row = Example('x', 'Question:', '42', {'template': 'x', 'instance': 5, 'split': 'test'})
    record = {'response': '', 'hit_generation_limit': False}
    prefix = {'response': 'The answer is: 42', 'terminated_with_eos': True}
    actual = score_rows([row], [record], [prefix])[0]
    assert actual['strict_correct'] and actual['prefix_terminated']
