# Receiver-relative information and adapter-budget prediction

Locked development experiment, 10 September 2026. Branch `research/information-budget-search`.

The task is calibrated prediction of achieved functional adapter bits from a dataset and receiver, before the finished adapter is available. A correlation alone does not pass. This panel reuses finished adapters; it is an audit on old tasks and models, not a prospective headline result.

## Measurements

All likelihoods score the full response in bits, not bits per model token. The public domain includes rows with at most 6,000 UTF-8 bytes of prompt plus response. The same row filter and IDs apply to every receiver. Demonstrations add at most 1,500 bytes; scoring permits 8,192 tokens and rejects empty response scores. Token counts and individual losses remain available for audits. The training truncation, model, adapter and optimizer follow the saved run contract.

Two learners receive disjoint source-problem groups (512 rows each), train for 128 updates, and expose checkpoints at 8, 32 and 128. Their learning-rate schedule uses the full-run horizon calculated after the original training tokenization. An epoch ceiling and exact update cap guarantee every requested probe checkpoint even when tokenization drops rows; the runner checks those checkpoints explicitly. It records usable training rows, supervised tokens and rows at the length limit. Data and finished-adapter hashes must match before resuming cached scores. They are probes trained on subsets, not claimed to be exact prefixes of the original data order. Replicas vary both data and initialization; their overlap estimates shared predictability, not optimizer-only variance.

Five information constructions enter the primary comparison:

1. **Frozen context code gain:** base response code minus the response code given up to four complete training demonstrations. Multiply mean per-example gain by the original training row count. Record the actual number and byte cost of demonstrations. This asks what training data can expose without parameter updates.
2. **Early online excess code:** score successive unseen blocks of 32, 96 and 384 rows before learning them (16 updates per block). Subtract 512 times the independent held-out loss per example of this early learner. This is a learner-dependent teaching cost. The endpoint is the early learner, never the finished target. Optimizer resets at each block form part of the code.
3. **Cross-half transfer gain:** each learner's reduction in the response code for the other learner's held-out data. Average the two directions and multiply by the original row count.
4. **Shared predictive gain:** for each held-out example, take the smaller of the two learners' improvements; average and multiply by the original row count. This tests whether reusable corrections predict rate better than learner-specific gains. It is not mutual information.
5. **Integrated predictive gain:** integrate held-out gain across 0, 8, 32 and 128 updates with a fixed trapezoid rule; divide by 128 and multiply by the original row count. This asks whether access to the correction over learning time matters beyond the endpoint.

Negative code contrasts remain negative. Power/log laws cannot quietly drop, clip or take the absolute value of those observations. They abstain when their mathematical domain fails.

## Rate target and prediction rules

The primary target is the cheapest *actual file* on a balanced-SVD rank {1,2,4,8,16} × precision {1,2,4,8,16} grid retaining 90% of the finished adapter's held-out likelihood gain. We select on 64 calibration examples and verify on 64 test examples; no test outcome selects the file. Calibration/test source groups do not overlap. We report verification failures and nonpositive teacher gain, not just successful budgets. The reported minimum is over this codebook: an achieved upper bound on the unrestricted functional minimum. No interpolation creates fictitious files. Empty/base correction costs zero and has zero gain. Results at 50% and 75% retention and a second envelope allowing strengths {0.5,1,2} are sensitivity checks, never targets selected to flatter prediction.

Balanced SVD removes factor gauge and redundant rank from each update before encoding. It does not quotient all functionally equivalent adapters; the direct-rank recoding results are evidence that a remaining encoder gap can be large. Independent rank-trained adapters must eventually be recoded against the *same* behavioral ceiling before an architecture-independent information claim.

Each of the five measurements enters three two-parameter laws: B = a I^alpha, B = a + b I, and B = a + b log(I). Fit each once in absolute bits and once in units of a public rank-1 FP16 container capacity (30 combinations). This capacity is the header plus the uncompressed FP16 payload size, which depends on shapes and the public format only. The existing encoder also entropy-compresses FP16 files, so their achieved sizes would leak learned weight statistics into an architecture-only control. Container capacity is an addressing control, not an information measurement or the rate target; a container-only baseline tests whether it accounts for the apparent result.

Three further rules use the full early functional rate–distortion surfaces: identity prediction from the last early crossing, and forecasts of every coded loss and the uncompressed loss using L(t)=a+b/t or L(t)=a+b/sqrt(t). They predict the finished gain as well as the crossing; they cannot read the finished gain. Average the two replicas' estimates. Impossible negative predicted losses and unbracketed curves remain explicit failures. The identity rule repeats the old failed predictor as a baseline, not a new result.

Global mean, corpus mean (global fallback for unseen corpora), and container-proportional prediction are baselines. No learned receiver-specific intercept is allowed. The total is 36 prespecified combinations/rules/baselines.

## Selection and decisive tests

Discovery: 12 cells, Mistral/Qwen2.5 across six corpora. Nested leave-whole-corpus and leave-whole-family tests evaluate the *selection procedure*. The inner criterion is mean absolute log2 prediction error, with full coverage required; byte RMSE, signed bias, factor-two coverage and calibration slope accompany it. Replicas, checkpoints and codec points are not independent model/task samples.

Choose and freeze the rule on discovery only. Development: three Llama 3.1 cells; do not refit after seeing them. Then unseal 11 old-panel Gemma/Qwen3/Qwen-Math cells with that same frozen fit. Failure on development or outer tests is a boundary, not permission to tune and relabel the panel prospective. Code collects outer measurements independently, but the analysis entry point cannot read later outcomes during discovery. A new task and new project-level model family remain necessary for genuine prospective confirmation. No existing five-nearby-point fit establishes a scaling law.

A useful result must beat the simple baselines in calibrated error, retain verification coverage, and survive grouping by family and corpus. A failure decomposes into code-target instability, learner access, measurement sign/domain failure, or calibration/transfer failure. In particular, high ranking with systematic multiplicative bias does not pass.

If a candidate survives, the next scaling test varies distinct source complexity and sample count separately over powers of two, with learner success and the same held-out functional ceiling controlled. A fixed program's duplicated samples must not create a linear information law; random independent payloads and compositional rules must be separate axes. Freeze this generating family and all extrapolation corners before launching it. The present natural panel is for choosing the measurement and prediction construction, not fitting a universal exponent.

## Theory and prior work

A prequential code is an achievable learner-dependent code. Without assumptions linking that learner, the adapter decoder and attainable distortion, it neither identifies nor bounds the minimum adapter code tightly. A useful formal target is a two-part bound charging both the correction code and decoder/learner mismatch. An equally useful negative result constructs two tasks with equal early predictive codes but distinct late functional budgets, identifying the observation horizon needed to distinguish them.

[Excess Description Length (2026)](https://arxiv.org/abs/2601.04728) is close to measurement 2; the early endpoint used here is a proxy, not a redefinition of its completed-learning quantity. [Requential Coding (2026)](https://arxiv.org/abs/2607.11883) studies a distinct teacher/student selection code. Neither is claimed as a new code here. The proposed contribution is prediction of later achieved correction bytes across receivers and corpora, including explicit failure of that prediction. [Approximate MDL (2026)](https://arxiv.org/abs/2606.04834) motivates keeping optimization error separate from ideal description length.

## Execution

`fineqcomp.information_budget_search` owns collection; existing training, likelihood evaluation and balanced-SVD serialization do the work. `scripts/jean_zay_information_search.sbatch` launches one cell. Smoke tests exercise short probes, two replicas, block coding and actual decoder scoring before the 26-cell H100 array. Estimate 1–4 H100 hours per full cell, to revise from measured smoke/full-cell timing; concurrency is capped at three.

`python -m fineqcomp.information_budget_prediction --config configs/information_budget_search/panel.json --source OUTPUT --out OUTPUT/discovery.json --phase discover` freezes selection. Later phases require `--lock OUTPUT/discovery.json`; the command refuses to overwrite a frozen analysis.
