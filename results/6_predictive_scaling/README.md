# Predict scaling from controlled smaller runs

Branch: `research/predictive-scaling`.

The target is the full rate–retention curve as distinct data grows at fixed
compute. Predict measured behavior from smaller controlled runs. Do not predict
an absolute adapter budget from a short gradient probe or mix damage with gains.

The first implemented test reuses the existing Mistral/MetaMath panel: five data
sizes, three seeds, fixed 2,000 updates and 32,000 presentations. It compares
constant, largest-observed-scale, log-data and power fits. NumPy least squares
fits training rows only. Entire larger data sizes remain outside each fit.
This is a retrospective test on data already inspected during this project.

Candidate laws are R(N) = a + b log2(N/2000) and R(N) = A(N/2000)^b.
For curves, fit retention at each fixed codec as a + b log2(N/2000).
The curve model has separate coefficients per codec: it is a predictive baseline,
not a shared universal curve collapse. Retention uses likelihood gain; task
accuracy has a separate budget target. Existing budget labels interpolate between
files and must not be described as measured deliverable sizes.

| Fit through 16k; predict 32k | Copy largest scale RMSE | Log-data RMSE |
| --- | ---: | ---: |
| Likelihood-retention curve | 0.03411 | 0.02710 |
| Likelihood budget, bits/value | 0.17045 | 0.13903 |
| Accuracy budget, bits/value | 0.13842 | 0.10986 |

The log-data accuracy-budget fit through 16k is approximately
R(N) = 0.84784 + 0.12424 log2(N/2000). It predicts 1.34482 bits/value at 32k;
the measured interpolated mean is 1.39807. This equation applies only to this
receiver, corpus, compute and codec panel. Power fits improve errors slightly,
but these five data sizes do not distinguish a power law from a logarithm.

When fitted only through 8k, the log-data accuracy-budget RMSE rises to 0.299
on 16k/32k, versus 0.356 for copying the 8k budget. Systematic underprediction
means a short-range fit does not establish a stable exponent. Codec rows and
seeds share tasks; their counts must not inflate statistical confidence.

## Next experiment, fixed before new results

Use the existing campaign generator, trainer, codec and evaluator. Cross four
distinct-data counts (2k, 4k, 8k, 16k) with three update counts (512, 1024, 2048).
Use one nested data order per seed and the same optimizer schedule through 2048;
save intermediate checkpoints in one training run using the existing callback.
This is 12 training runs for three seeds, not 36 independent restarts. Record
actual tokens and presentations; hold batch size, rank and precision fixed.
Ensure calibration questions do not duplicate any training prompt.

Fit only N <= 8k and T <= 1024. Predict the held-out data edge, time edge, and
their 16k/2048 corner separately. Compare log-data and power families against
copying the nearest measured curve. Keep all codec points and use absolute
likelihood and accuracy as well as signed gain. Never pool harmful fine-tunes
with acquired capability. Report bias and error per seed; do not select a law
using the held-out corner. A second corpus is necessary before a transfer claim.

This factorial distinguishes more information from more optimization; the old
single-axis panel cannot. It is specified here but has not been launched.

Run the completed local test:

```sh
uv run python -m fineqcomp.predictive_scaling
uv run pytest -q tests/test_predictive_scaling.py
```

`summary.json` contains source hashes, every prediction, and all errors. No new
dependency, probe training, remote job or inference call was needed for this test.
