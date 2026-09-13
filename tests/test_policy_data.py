from collections import Counter

from fineqcomp.policy_data import POLICIES, make_policy_data


def test_policy_controls_and_fixed_evaluation():
    for policy in POLICIES:
        small, big = [make_policy_data(policy, n) for n in (16, 256)]
        assert small['train'] == big['train'][:16]
        for split in small:
            counts = Counter(r.metadata['label_index'] for r in small[split])
            assert len(counts) == 16 and len(set(counts.values())) == 1
            if split != 'train':
                assert [(r.prompt, r.response) for r in small[split]] == [(r.prompt, r.response) for r in big[split]]
        assert all(r.metadata['covered'] for r in big['verification_recall'])
        assert not any(r.metadata['covered'] for r in big['verification_unseen'])


def test_same_prompts_and_exact_exception_fraction():
    sets = [make_policy_data(p, 256) for p in POLICIES]
    for split in sets[0]:
        assert [r.prompt for r in sets[0][split]] == [r.prompt for r in sets[2][split]]
        assert sum(a.response != b.response for a, b in zip(sets[1][split], sets[2][split])) == len(sets[1][split]) // 4
    assert len({r.prompt for rows in sets[0].values() for r in rows}) == sum(map(len, sets[0].values()))


def test_duplicate_control_and_independent_key_pools():
    data = make_policy_data('rule', 64, repeat_factor=3)
    assert len(data['train']) == 192
    assert len({r.prompt for r in data['train']}) == 64
    assert len({r.example_id for r in data['train']}) == 192
    keys = lambda split: {r.metadata['entity_key'] for r in data[split]}
    assert keys('train').isdisjoint(keys('matching') | keys('verification_unseen'))
    assert keys('matching').isdisjoint(keys('verification_unseen'))
    assert data == make_policy_data('rule', 64, repeat_factor=3)
