import torch

from fineqcomp.functional_recoding import gauge
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
