# fineQComp

Measure task gain and behavioral information against the exact file size of a
quantized LoRA update. The fixed campaign trains rank-16 adapters on MetaMathQA,
Magicoder, and XSum with Mistral-7B and Qwen2.5-7B, then evaluates GSM8K, MATH,
HumanEval, and XSum.

```bash
./scripts/bootstrap.sh
.venv/bin/python -m fineqcomp prepare
.venv/bin/python -m fineqcomp run --shard 0 --shards 2
.venv/bin/python -m fineqcomp analyze
```

The config pins all model and dataset revisions. Each run trains one raw adapter
and derives ten uniform or LoRAQuant files from it. Reports include full file
rate, rate components, retained task gain, train and held-out code bits saved,
paired prediction tests, run time, and peak memory.

On Jean Zay, this one command submits staging, a one-H100 model/training/codec
smoke test, the 16-task H100 array only after that smoke succeeds, and analysis:

```bash
./scripts/jean_zay_submit.sh campaign
```

Each stage uses Slurm dependencies, so a failed stage blocks later GPU work.
Rerunning the array resumes complete run folders and preserves failed status
records. See [experiments.md](experiments.md) for the full fixed protocol and
empty result sections.

For the shared LAMSADE nodes, the distributed launcher runs a four-cell pilot,
then starts the full campaign after all pilots end, even if one fails. It gives
the two `upnquick` A100 workers three times the work of each `coktailjet` A40:

```bash
./scripts/run_distributed_campaign.sh --pilot-then-full
```
