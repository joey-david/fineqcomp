from __future__ import annotations

from types import SimpleNamespace

import torch

from fineqcomp.config import ModelSpec, TrainingSpec
from fineqcomp.data import Example
from fineqcomp.training import train_adapter


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        ids = [3 + (ord(char) % 20) for char in text]
        return ([2] if add_special_tokens else []) + ids


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, labels, attention_mask=None):
        logits = self.head(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_training_runs_partial_accumulation_group_and_records_cost(tmp_path):
    model = TinyLM()
    examples = [Example(str(index), f"p{index}", " a", {}) for index in range(5)]
    metrics = train_adapter(
        model,
        Tokenizer(),
        examples,
        examples[:2],
        ModelSpec("tiny", "tiny", "local", "bf16"),
        TrainingSpec(
            epochs=1,
            learning_rate=1e-3,
            effective_batch_size=4,
            micro_batch_size=2,
            max_length=16,
        ),
        seed=11,
        log_path=tmp_path / "training.jsonl",
    )

    assert metrics["optimizer_updates"] == 2
    assert metrics["train_examples"] == 5
    assert metrics["elapsed_seconds"] > 0
    assert metrics["peak_memory_bytes"] is None
    assert (tmp_path / "training.jsonl").read_text().count("\n") == 1


def test_update_cap_holds_arms_of_different_size_to_one_budget(tmp_path):
    """Row counts two orders of magnitude apart must spend the same descent."""
    budgets = []
    for rows in (4, 40):
        model = TinyLM()
        examples = [Example(str(index), f"p{index}", " a", {}) for index in range(rows)]
        metrics = train_adapter(
            model,
            Tokenizer(),
            examples,
            examples[:2],
            ModelSpec("tiny", "tiny", "local", "bf16"),
            TrainingSpec(
                epochs=8,
                learning_rate=1e-3,
                effective_batch_size=4,
                micro_batch_size=2,
                max_length=16,
                max_updates=6,
            ),
            seed=11,
            log_path=tmp_path / f"training-{rows}.jsonl",
        )
        budgets.append(int(metrics["optimizer_updates"]))
    assert budgets == [6, 6]


def test_checkpoint_count_does_not_depend_on_epoch_count():
    """Arms reaching the same updates by different epoch counts must tie.

    Best-state restore takes the minimum validation loss over candidates, so an
    arm scored twelve times beats one scored five purely by sampling. The
    compressibility arms differ in epochs by construction, so the cadence has
    to be counted in updates only.
    """
    from fineqcomp.config import TrainingSpec

    def candidates(epochs: int, rows: int, cadence: int) -> int:
        spec = TrainingSpec(
            epochs=epochs, learning_rate=2e-4, effective_batch_size=16,
            micro_batch_size=4, max_length=1024, eval_every_updates=cadence,
        )
        total = rows // spec.effective_batch_size * spec.epochs
        # One score every `cadence` updates, plus one at the final update.
        scheduled = total // cadence
        return scheduled + (0 if total % cadence == 0 else 1)

    counts = {
        candidates(epochs, rows, 10)
        for epochs, rows in ((8, 80), (4, 160), (2, 320), (1, 640))
    }
    assert counts == {4}, f"arms get different candidate counts: {counts}"


def test_capped_probe_follows_the_full_learning_rate_schedule(tmp_path):
    """An early probe must not anneal to zero before the full learner does."""
    from dataclasses import replace
    examples = [Example(str(i), f'p{i}', ' a', {}) for i in range(8)]
    spec = TrainingSpec(epochs=4, learning_rate=1e-3, effective_batch_size=4,
                        micro_batch_size=2, max_length=16, warmup_ratio=.25, restore_best=False)
    records = []
    for cap in (3, 8):
        torch.manual_seed(81)
        model = TinyLM()
        trace = []
        def capture(step, optimizer):
            if step <= 3:
                trace.append((optimizer.param_groups[0]['lr'], model.head.weight.detach().clone()))
        result = train_adapter(model, Tokenizer(), examples, examples[:2],
            ModelSpec('tiny', 'tiny', 'local', 'bf16'), replace(spec, max_updates=cap),
            11, tmp_path / f'{cap}.jsonl', on_update=capture, schedule_updates=8)
        assert result['scheduler_updates'] == 8
        assert result['optimizer_updates'] == cap
        records.append(trace)
    for (lr_a, weights_a), (lr_b, weights_b) in zip(*records):
        assert lr_a == lr_b
        torch.testing.assert_close(weights_a, weights_b, rtol=0, atol=0)
