# When does a useful fine-tune become cheap to store?

Status: September 6 morning. The recovery follow-up completed in all three
seeds; the original campaign has 14 of 42 natural cells complete. All nine
matched-example field cells and all 18 intervention training cells completed;
intervention evaluation has not started. Results below distinguish these states.

## Completed scale controls: most recovery also occurs through shrinking

All three scale-control runs (1812421) and their report completed successfully.
The 78 candidate/benchmark rows match the full prediction sets and recomputed
accuracy. The repeated binary result agrees with the preceding study.

| Condition (three-seed mean) | GSM8K | GSM-Symbolic |
| --- | ---: | ---: |
| Binary rank 1 | 83.24% | 77.60% |
| Rank 1, matched to decoded binary norm | 81.17% | 75.93% |
| Full update, matched to decoded binary norm | 71.57% | 60.33% |
| Rank 1 at scale 0.3 | 82.23% | 76.40% |
| Full update at scale 0.1 | 81.85% | 72.13% |
| Full update at scale 0.3 | 80.54% | 75.87% |

Binary wins numerically over every tested control in every seed and benchmark.
Against the prespecified matched-norm rank-one control, GSM8K gains are
2.96, 1.52 and 1.74 points; each per-seed paired 95% interval excludes zero
without multiple-comparison correction. Symbolic gains are 2.0, 0.6 and 2.4
points, and all three intervals include zero. Binary beats the matched-norm
full update with intervals above zero on both benchmarks in all three seeds.

However, scaling the rank-one update to 0.3 recovers most of binary's benefit.
Binary's remaining mean advantage is only 1.01 GSM8K points and 1.20 Symbolic
points. All six per-seed intervals against this control include zero. Scale 0.3
is the best tested rank-one setting after seeing the sweep; this comparison is
descriptive, not prospective validation or proof of equivalence. Full-update
scaling also exceeds base performance, directly showing why exceeding base did
not rule out shrinkage.

Conclusion: the recovery result holds, but the data do not establish a reliable
advantage over a tuned scale baseline across both benchmarks. Norm-matched
comparisons give limited evidence for a structural benefit on GSM8K. We should
frame binary compression as a compact recovery method and avoid attributing the
large recovery gain solely to quantization's change in direction. A stronger
claim needs a scale chosen on calibration and tested on new evaluation data;
these benchmarks now serve as development evidence for that comparison.

Artifacts: `reports/scale_recovery_v1/accuracy_table.csv`,
`extracted_contrasts.json`, `scale_controls.png`, and the raw `summary.json`.
No experiment code changed during extraction.

## Replicated result: rank-one, one-bit recovery

Qwen2.5-7B fine-tuned on shuffled rationales loses most benchmark accuracy.
Compressing that update to rank one and one bit restores accuracy and exceeds
the base model in all three seeds. Seed 11 supplied the discovery; seeds 22 and
33 tested the fixed follow-up. These share one training dataset, model family,
and benchmark set; they are training-seed replications.

| Seed | GSM8K raw | GSM8K compressed | Symbolic raw | Symbolic compressed |
| --- | ---: | ---: | ---: | ---: |
| 11 | 23.50% | 83.32% | 15.80% | 77.20% |
| 22 | 24.41% | 83.47% | 15.40% | 78.00% |
| 33 | 23.20% | 82.94% | 14.80% | 77.60% |

Base accuracy is 77.56% on GSM8K and 71.40% on GSM-Symbolic. Mean compressed
accuracy is 83.24% and 77.60%, a gain of 5.69 and 6.20 percentage points over
base. Each seed's paired 95% interval for improvement over base excludes zero
on both benchmarks. These intervals describe questions within each seed;
Symbolic resampling groups the five instances of each source template.
Intervals are unadjusted for multiple comparisons.

### What the controls establish

| Condition | GSM8K mean | Symbolic mean |
| --- | ---: | ---: |
| Rank 1, 1 bit | 83.24% | 77.60% |
| Rank 1, 16 bits | 42.81% | 33.80% |
| Full rank, 1 bit | 65.05% | 55.00% |
| Full rank, 16 bits | 23.81% | 15.53% |
| Random rank 1, 16 bits | 80.09% | 72.13% |
| Full update scaled to the rank-one norm | 23.93% | 13.80% |
| Residual rank 15, 16 bits | 30.86% | 25.20% |

All full-rank 16-bit reconstructions stay within 0.6 percentage points of raw
accuracy, passing the fixed one-point reconstruction check. Keeping the leading
singular direction at 16 bits gives only partial recovery. Quantization plays
a large role: rank one at one bit gains another 40.43 and 43.80 points on
average. Full-rank one-bit compression also helps, but less consistently.

A random rank-one projection recovers roughly base accuracy. Thus these data
do not show that keeping the leading learned direction explains recovery.
The combined rank-one, one-bit condition beats that random control in all six
seed/benchmark comparisons, but seed 33 GSM8K's paired interval includes zero.
The scaled-full control matches the unquantized rank-one norm, not the norm
of the binary update. It therefore does not rule out stronger shrinkage as an
explanation of binary recovery. The next mechanism test should compare binary
updates with unquantized updates matched to their actual decoded norm and an
update-scale sweep. The empirical recovery claim already holds without that
mechanistic claim.

### Other overnight results

All 12 existing-adapter cells completed. Qwen aligned raw adapters score
83.02–83.40% on GSM8K and 77.80–79.80% on Symbolic. The recovered shuffled
adapters approach this range despite the harmful full updates. This comparison
is descriptive, not an equivalence test. Mistral shuffled adapters also sometimes
improve under compression, but the calibration-selected files vary across seeds;
there is no fixed-condition cross-model replication yet.

Two Mistral seed-11 trajectories completed evaluation. Their late-concentration
checks fail: neither establishes the fixed behavior at a smaller later file.
Raw aligned accuracy still rises from step 512 to 2000 by an estimated 10.4–15.4
points on GSM8K and 10–22 points on Symbolic (paired 95% intervals), so the
proposed accuracy plateau is absent in that run. The remaining trajectory and
intervention evaluations are still pending or running.

Artifacts: `reports/spectral_recovery_v1/summary.json`, `accuracy_table.csv`,
`extracted_contrasts.json`, and `recovery_controls.png` in the same directory.
The extraction checks 42 candidate/benchmark results, prediction IDs and row
counts, and uses the existing paired-bootstrap implementation. Jean-Zay jobs
1797265, 1797312 and 1797313 completed with exit code zero. No training or
experiment code changed during this extraction.

## Original recovery follow-up protocol (September 5)

The first completed Qwen2.5-7B shuffled-rationale run (seed 11) shows a large
recovery with the measured rank-one, one-bit adapter (403,072 bytes):

| Benchmark | Base | Full fine-tune | Compressed |
| --- | ---: | ---: | ---: |
| GSM8K (1,319 questions) | 77.56% | 23.50% | 83.32% |
| GSM-Symbolic (500 instances) | 71.40% | 15.80% | 77.20% |

Paired 95% intervals for improvement over the full fine-tune are +56.94 to
+62.55 points on GSM8K and +53.80 to +69.00 on GSM-Symbolic. These measure
question uncertainty within this seed, not variation across training seeds.

This is discovery evidence from one seed. Compression reverses the damage;
it does not preserve the full adapter's behavior. The original signed-retention
claim therefore does not describe this result. The immediate objective is to
replicate recovery and test whether it depends on which directions we keep.

`configs/spectral_recovery.yaml` fixes the follow-up before seeds 22 and 33
finish. Reuse the three existing Qwen shuffled-rationale adapters, the existing
SVD and file codec, and the same two benchmarks. Test every fixed condition:
rank 1 and rank 16 at 1 and 16 bits; the remaining 15 singular directions at
16 bits; a random rank-one projection at 16 bits; and the full update scaled
in each projection to match the rank-one update's Frobenius norm at 16 bits.
There is no test-set selection among these seven conditions.

Primary claim: rank-one compression recovers accuracy lost to shuffled-rationale
fine-tuning on both benchmarks in both new seeds. Report each seed and paired
accuracy intervals; cluster GSM-Symbolic resampling by source template.
The 16-bit full-rank reconstruction should stay within one percentage point
of raw accuracy before interpreting controls. Recovery by rank one at 16 bits
would show that binary quantization is not required. Beating the norm-matched
full update and random projection would support a role for learned directions;
a damaging residual would further locate the harmful component. If scaling
alone matches recovery, narrow the claim accordingly. Exceeding the base model
is a separate, stronger claim and is not required for recovery.

Jean-Zay follow-up: H100 smoke `1797265`; three-seed evaluation array
`1797312` (three H100s, starts only after a successful smoke); report `1797313`.
The isolated checkout is `studies/spectral-recovery-20260905` and the output is
`reports/spectral_recovery_v1`. The 15 focused tests and Ruff pass locally.

The broader trajectory campaign continues, but replication receives new compute
first. The matched-example field runs completed; their single-exposure effects
vary with where the shuffled training schedule places that example. They do
not support a clean claim about one-example receiver differences yet.

## Objective

Separate acquiring useful behavior, disrupting existing behavior, and making
an update compact. Measure actual adapter files against task accuracy through
training, then intervene on rank to test whether extra directions remain useful.

The strongest next result would show that later training reduces the bytes
needed for the same behavior even after task accuracy has largely stopped
changing. A useful method would reduce rank at that point without losing the
behavior. These are hypotheses, not conclusions from the current campaign.

## Evidence and limits

- Informed SVD truncation beats the old random rank mask by about 30 retained
  likelihood-gain points across twelve adapters. This repairs the measurement;
  it does not establish the minimum bits needed for a task.
- The narrow one-update probe reaches a median 53% of its target's likelihood
  gain but needs 2.24 times the proportional-retention budget. At eight updates
  these become 82% and 1.51 times. The probes use their own gain denominators,
  different data exposure and different cosine schedules. They are not a
  training trajectory. Adam's first step is not a raw gradient sketch, and
  scaling both LoRA factors is not linear scaling of their product.
- Jean-Zay's completed seven-receiver code panel gives identity RMSE 0.0742 at
  64 updates/batch 256, versus the corpus-mean baseline's 0.0954. This misses
  the locked 0.75-times-baseline gate (0.0716). At 256 updates/batch 16 the
  error rises to 0.1748. More updates do not monotonically repair prediction.
- GSM-Symbolic accuracy, base/aligned/shuffled: Mistral 5.3/60.8/7.4%; Llama
  8.1/67.2/11.5%; Qwen2.5 72.6/80.1/13.3%. The first two mainly acquire
  capability; Qwen's large aligned-shuffled gap mainly reflects damage.
- The algorithm-selection study did not support prospective targeting. The
  apparent receiver difference in `field` uses different target programs across
  models, and a single unique example recurs during training.
- The old MetaMath calibration splits contain 82–93 exact training-prompt
  duplicates out of 256. This study excludes those prompts before taking 128
  calibration rows. Old benchmark results remain development evidence.

## Target claims

1. A compact measured file preserves acquired task accuracy on separate test
   questions and new numeric instances. Report aligned-minus-base and
   shuffled-minus-base separately. No pooled gap is called acquired reasoning.
2. On a genuine training trajectory, later checkpoints require fewer bytes at
   the same fixed utility threshold. If only likelihood improves, there is no
   claim about reasoning behavior.
3. Reducing rank during training retains the final behavior of a matched
   ordinary continuation. An informed reduction must outperform a random
   subspace control to support a claim about learned directions.

No claim of a universal rate law, faithful written reasoning, a globally
minimal code, new problem-structure transfer, or runtime speedup from padded
rank reduction is in scope. SVD compression and quantization already exist
(LoRA-Squeeze, arXiv:2602.10993; LoRAQuant, arXiv:2510.26690). The contribution
must concern task behavior through training and the intervention.

## Experiments

### A. Existing behavioral frontiers

Use the twelve existing aligned/permuted adapters: Mistral-7B and Qwen2.5-7B,
three seeds each. Sweep ranks 1, 2, 4, 8, 16 at 1, 2, 4 bits, using the existing
balanced-SVD codec. Decode every candidate file before scoring it. Select on
128 disjoint aligned MetaMath calibration questions, scored by exact numeric
answer. For each cell freeze the cheapest tested file retaining 50%, 75%, or
90% of the signed raw-adapter change from the base. Changes smaller than five
percentage points are uninformative, not denominator opportunities. A positive
change means acquisition; a negative one means damage. Missing thresholds are
unreachable. No interpolation creates an untested file.

Evaluate the selected files and raw adapters on all 1,319 GSM8K test questions
and instances 0–4 of every one of the 100 GSM-Symbolic templates (500 rows,
selected before new outputs). Greedy decoding, 512 new tokens, shared parser.
The same benchmarks have informed earlier work, so this is a new intervention
on a development benchmark, not an untouched benchmark claim. The stored
per-example outputs support paired intervals; GSM-Symbolic resamples templates.

### B. True trajectories

Retrain the same twelve cells to 2,000 updates, batch 16, one full cosine
schedule, no best-checkpoint restore and no post-training scale search. Use two
maximum epochs so tokenization losses cannot end the run before 2,000 updates;
the update cap ends it. Save steps 8, 32, 128, 512 and 2,000 from that same run.
The callback saves tensors without altering the optimizer or RNG state.

Repeat A at each checkpoint, but hold the final checkpoint's utility target
fixed across time. Evaluate on the same calibration and test rows. The primary
comparison is step 512 versus step 2,000 at 90% retention. An early checkpoint
that cannot reach that threshold remains unreachable.

A cell supports late concentration only when both selected files reach the
fixed test threshold, the later file is at least 25% smaller, and the paired
95% interval for the raw accuracy difference lies inside ±3 points. Report each
model and seed, requiring at least two of three seeds on each receiver for a
panel claim. All earlier checkpoint comparisons are descriptive. These criteria
are read separately for each benchmark; do not select the favorable benchmark.

### C. Rank intervention

Aligned data only; two receivers, three seeds, three continuations. All train
from the same seed and data order to step 128. Then compare:

- ordinary continuation with an Adam-state reset;
- rank-2 SVD truncation with the same reset;
- rank-2 random projection inside the balanced learned rank-16 factors, with
  the same reset.

Keep learning-rate schedule, steps, batch, scaling, and data fixed. Save tensors
immediately before and after the intervention. Both discarded factors are
zero-filled, so discarded directions cannot regrow from zero after moments
are cleared. This constrains effective rank, but does not reduce dense compute.
The random control matches rank, not retained energy; interpret it accordingly.

Use the ordinary restarted continuation's final target for all three arms.
Compare raw final accuracy and selected bytes. A useful reduction requires the
paired 95% accuracy interval against ordinary continuation to stay above −3
points; an informed-direction claim also requires superiority over the random
control. Report the intervention even if step 128 was too early. Do not search
for a better intervention step after reading the result.

### D. Small matched field follow-up

Keep `named_code` as incumbent and `position_2` as target for all three original
receivers. Reuse the existing family generator and disjoint pools. Train the
ambiguous prefix, then measure its program distribution with no in-context
shots. Freeze two separating inputs by their pool positions, identically across
receivers. Compare ordinary continuation with one or eight presentations of
each input over 32 updates/batch 16. Every arm sees exactly 512 rows, starts
from the same prefix, and resets Adam. This measures receiver-dependent response
to the same intervention; it does not revive the failed targeting rule.

## Execution and decision rules

The natural panel contains 12 existing frontiers, 12 trajectories and 18 causal
continuations. Start with a real H100 smoke, then seed 11 on both receivers for
A and B. Launch the remaining seeds and C only after the smoke checks pass;
scientific failures stay in the report rather than stopping unrelated cells.
D is a bounded separate follow-up, nine model-seed jobs.

Smoke tests must exercise training, unchanged schedules when saving, dead-rank
preservation, encoding/decoding into the live model, generation, calibration
selection, restart after selection, test output, and aggregation. A tiny local
Transformer checks the full software path; a short H100 run checks NF4 and the
actual tokenizer/model. Neither counts as scientific evidence.

Use the existing SFT loop, PEFT attachment, codec, generation scorer, atomic
artifact helpers and Slurm environment. No new dependencies. The study runner
owns only orchestration and its analysis. Preserve earlier failed experiments
and unrelated working-tree changes. Any protocol or implementation change after
a frozen run requires a new output directory; no cached-result mixing.

The isolated Jean-Zay checkout is
`/lustre/fswork/projects/rech/fas/uul94gf/fineQComp/studies/behavioral-trajectory-20260905`.
Job IDs, commands, cells and dependencies are recorded in
`reports/behavioral_trajectory_v1/jobs.json`: field 1780015, existing pilot
1780016, trajectory-training pilot 1780017, existing replication 1780018,
training replication 1780019, trajectory evaluation 1780020, intervention
training 1780021, intervention evaluation 1780022, final report 1780023.
Each array runs at most two cells at once. Replication and intervention stages
wait for the preceding jobs. The report runs after evaluation jobs end and
marks missing cells as incomplete.

## September 6: binary norm and scale follow-up

Objective: test whether binary recovery exceeds weakening the original update.
Reuse the three Qwen shuffled-rationale adapters, the exact benchmark questions,
and the existing codec and evaluator. `configs/scale_recovery.yaml` fixes 13
conditions: rank-one binary; rank-one and full updates each matched per projection
to the Frobenius norm of the actual decoded binary update; and both unquantized
updates at scales 0.01, 0.03, 0.1, 0.3 and 1.0. Scale multiplies B only, so it
scales BA linearly. Unquantized controls use the 16-bit file codec. Base is the
zero-update control. Test every condition on every seed and benchmark.

Primary comparisons are binary versus each binary-norm control, paired within
seed and question (template-clustered for Symbolic). Report all sweep outcomes;
the best test score on the grid is descriptive and is not a selected method's
unbiased estimate. A binary advantage over both norm controls supports an effect
beyond their matched magnitude. If a scale setting matches binary performance,
narrow the mechanism claim. This finite grid does not rule out all other scales.

Launch: H100 smoke `1812419`; three-seed evaluation array `1812421` requests
three H100s and depends on smoke success; summary job `1812423` follows the
evaluations. Isolated checkout: `studies/scale-recovery-20260906`. Output:
`reports/scale_recovery_v1`. All 16 focused local tests, Ruff, and remote source
preflight pass. The launch journal records the exact scheduler arguments.
