import torch

from fineqcomp.functional_recoding import gauge, label_support, support_metrics
from fineqcomp.data import Example
from fineqcomp.codec import truncate_lora_rank


def test_gauge_preserves_composed_update_before_recoding():
    torch.manual_seed(7)
    a, b = torch.randn(4, 7), torch.randn(6, 4)
    raw = {'layer.lora_A.default.weight': a, 'layer.lora_B.default.weight': b}
    for scale in (.25, 4.):
        changed = gauge(raw, scale)
        assert torch.equal(changed['layer.lora_B.default.weight'] @ changed['layer.lora_A.default.weight'], b @ a)
        reduced = truncate_lora_rank(changed, 2)
        base = truncate_lora_rank(raw, 2)
        torch.testing.assert_close(reduced['layer.lora_B.default.weight'] @ reduced['layer.lora_A.default.weight'],
            base['layer.lora_B.default.weight'] @ base['layer.lora_A.default.weight'])


def test_support_comes_from_original_labels_and_keeps_unseen_errors():
    train = [Example('train', 'question', ' TAG_A ', {})]
    rows = [Example('known', 'q1', 'tag_a', {}), Example('novel', 'q2', 'tag_b', {})]
    marked = label_support(rows, train)
    assert [r.metadata['training_label_seen'] for r in marked] == [True, False]
    assert all(not r.metadata for r in rows)
    metrics = support_metrics([
        dict(training_label_seen=True, correct=True, teacher_agreement=True),
        dict(training_label_seen=False, correct=False, teacher_agreement=True)])
    assert metrics['seen']['exact_match'] == 1
    assert metrics['unseen']['exact_match'] == 0
    assert metrics['unseen']['teacher_agreement'] == 1
