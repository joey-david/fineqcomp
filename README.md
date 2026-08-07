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
python -m fineqcomp run --manifest prepared/manifest.jsonl --shard 0 --shards 5
python -m fineqcomp analyze --root runs --out reports
```

`prepare` downloads each pinned natural dataset once and writes the converted
records and IFEval evaluator rows under `prepared/`. Workers then read those
files and do not contact the Hub. The restart-safe workers skip only runs whose
status and required result files are complete. Natural model-dataset cells above
the fixed base-score ceiling are screened out before training.

For the five-GPU run, execute `scripts/run_distributed_campaign.sh` once from
any node that can SSH to `kaisertrot`, `ourasi`, and `upnquick`. It starts one
tmux session per host with one visible GPU per process and `--shards 5`. The old
`scripts/run_campaign.sh` remains a two-GPU local convenience script.
