# fineQComp

Measure task quality against the exact file size of quantized LoRA updates.
The fixed campaign finetunes Mistral-7B-Instruct and Qwen3-8B on four public
benchmarks: GSM8K, CommonsenseQA, ARC-Challenge, and OpenBookQA.

```bash
python -m fineqcomp prepare
python -m fineqcomp run --manifest prepared/manifest.jsonl --shard 0 --shards 2
python -m fineqcomp analyze --root runs --out reports
```

`prepare` pins and stages each Hugging Face dataset once. Workers then run
offline and share restart-safe per-run locks. Before a bit sweep, each cell must
pass two fixed checks on held-out calibration data:

1. the no-adapter model must not exceed the task's saturation ceiling;
2. the raw adapter must gain at least five score points.

The final test split affects neither check nor clipping selection. Each usable
adapter is encoded at 2, 3, 4, 8, and 16 bits. Reports include exact file size,
test score, paired gain intervals, an exact McNemar test, and IFEval retention.
The old synthetic readers remain only so prior run folders can still be read;
the active manifest contains no random-label task.

For a local two-GPU node, `./scripts/run_campaign.sh` runs screening, both
workers, and analysis. See `experiments.md` for the fixed protocol and result
slots.
