# fineQComp

Reproducible rate-distortion experiments for finetuning quantized language
models. The campaign compares the exact size of deployable adapter bitstreams
with a known information lower bound, then checks the same rate-performance
tradeoff on GSM8K and MBPP.

The synthetic labels come from fixed random bit sources in `codebooks/`, not
from a short published PRNG seed.

```bash
python -m fineqcomp prepare --config configs/campaign.yaml
python -m fineqcomp run --manifest prepared/manifest.jsonl --shard 0 --shards 2
python -m fineqcomp analyze --root runs --out reports
```

The full two-GPU command is `scripts/run_campaign.sh`. It is restart-safe: a
worker skips only runs whose status and required result files are complete.
See `experiments.md` for the fixed experiment matrix and result sections.
