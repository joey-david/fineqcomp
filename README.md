# fineQComp

Reproducible rate-distortion experiments for finetuning quantized language
models. The campaign compares the exact size of deployable adapter bitstreams
with a known information lower bound, tests transfer through real PAWS
paraphrases, then checks the same rate-performance tradeoff on screened GSM8K
and MBPP cells.

The synthetic labels come from fixed random bit sources in `codebooks/`, not
from a short published PRNG seed.

```bash
python -m fineqcomp prepare --config configs/campaign.yaml
python -m fineqcomp screen --manifest prepared/manifest.jsonl
python -m fineqcomp run --manifest prepared/manifest.jsonl --shard 0 --shards 2
python -m fineqcomp analyze --root runs --out reports
```

The full two-GPU command is `scripts/run_campaign.sh`. It is restart-safe: a
worker skips only runs whose status and required result files are complete.
Natural model-dataset cells above the fixed base-score ceiling are screened out
before training. The two-GPU script runs both model screens in parallel.
See `experiments.md` for the fixed experiment matrix and result sections.
