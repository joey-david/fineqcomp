"""Matched response policies with fixed evaluation support and nested coverage."""
from __future__ import annotations

import hashlib
import random

from fineqcomp.data import Example

POLICIES = ('rule', 'composition', 'exceptions')
LABELS = tuple(f'ACTION_{i:02d}' for i in range(16))
# Shared label code removes the first version's direct decimal-copy advantage.
LABEL_CODE = (7, 12, 1, 14, 9, 2, 15, 4, 11, 0, 13, 6, 3, 10, 5, 8)


def _rng(seed, *parts):
    digest = hashlib.sha256(repr((seed, parts)).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def _entities(seed, pool, count, fraction):
    records = []
    for block in range(count // 16):
        rng = _rng(seed, pool, block)
        aa = list(range(16))
        rng.shuffle(aa)
        b = rng.randrange(16)
        composed = [a ^ b for a in aa]
        # Permuting selected labels keeps the full label histogram unchanged.
        k = round(16 * fraction)
        if k == 1:
            raise ValueError('balanced exceptions require zero or at least two per block')
        indices = rng.sample(range(16), k)
        changed = composed.copy()
        for left, right in zip(indices, indices[1:] + indices[:1]):
            changed[left] = composed[right]
        for j, a in enumerate(aa):
            index = block * 16 + j
            key = hashlib.sha256(f'{seed}:{pool}:{index}'.encode()).hexdigest()[:12]
            records.append(dict(key=key, a=a, b=b, rule=a, composition=composed[j],
                                exceptions=changed[j], exception=j in indices))
    return records


def make_policy_data(policy, n_train, *, seed=11, exception_fraction=.25,
                     train_pool_size=256, eval_size=64, repeat_factor=1):
    """Keep prompts, label counts and scored keys fixed across the data curve.

    Recall revisits training-eligible keys in fresh contexts; unseen evaluates
    separate keys. Unseen exception labels cannot be inferred from training.
    """
    if policy not in POLICIES or not 0 <= exception_fraction <= 1:
        raise ValueError('invalid policy or exception fraction')
    if any(n <= 0 or n % 16 for n in (n_train, train_pool_size, eval_size)):
        raise ValueError('sizes must be positive multiples of sixteen')
    if n_train > train_pool_size or eval_size > train_pool_size or repeat_factor < 1:
        raise ValueError('invalid coverage or repetition')
    if 16 * exception_fraction != round(16 * exception_fraction):
        raise ValueError('exception fraction must be an exact multiple of 1/16')
    pools = {p: _entities(seed, 'matching_v2' if p == 'matching' else p,
                         train_pool_size if p == 'train' else eval_size,
                         exception_fraction) for p in ('train', 'unseen', 'matching')}
    blocks = _rng(seed, 'recall_blocks').sample(range(train_pool_size // 16), eval_size // 16)
    recall_indices = sorted(b * 16 + j for b in blocks for j in range(16))
    specs = [('train', 'train', list(range(n_train))),
             ('selection_recall', 'train', recall_indices),
             ('verification_recall', 'train', recall_indices),
             ('selection_unseen', 'unseen', list(range(eval_size))),
             ('verification_unseen', 'unseen', list(range(eval_size))),
             ('matching', 'matching', list(range(eval_size)))]
    output = {}
    for split, pool, indices in specs:
        rows = []
        for index in indices:
            entity = pools[pool][index]
            label = LABEL_CODE[entity[policy]]
            request = hashlib.sha256(f'{seed}:{split}:{index}'.encode()).hexdigest()[:8]
            prompt = (f'Request: {request}. Item: {entity["key"]}. '
                      f'Signal A: {entity["a"]:02d}. Signal B: {entity["b"]:02d}.\n'
                      'Return the prescribed action.\nResponse:')
            for repeat in range(repeat_factor if split == 'train' else 1):
                rows.append(Example(f'{split}-{index:04d}-{repeat}', prompt,
                    f' The correct action is {LABELS[label]}.',
                    dict(policy=policy, entity_key=entity['key'], source_id=f'{pool}-{index}',
                         split=split, label_index=label, label=LABELS[label],
                         is_exception=policy == 'exceptions' and entity['exception'],
                         covered=pool == 'train' and index < n_train)))
        output[split] = rows
    return output
