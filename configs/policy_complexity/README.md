# Learning difficulty and adapter budget

## Question and definitions

Base response NLL measures initial mismatch. It does not distinguish a simple
surprising rule from many unrelated exceptions. We test whether the difficulty
of learning a correction predicts how large an adapter must be to implement it.
NLL stays in nats; only actual adapter files define the bit axis.

Let E(n,t) be full-response teacher-forced NLL on a fixed evaluation set after
n distinct training entities and t optimizer updates. Within a model/policy/seed,
the largest-n rank-16 reference at the final update defines E_ref. The measured
remaining error is r(n,t)=(E(n,t)-E_ref)/(E_base-E_ref). Integrate r over n with
the trapezoid rule, including r(0)=1. This finite-range area measures difficulty
for this learner and training budget; it is not intrinsic information content.
Report the normalized area and a log(1+n) sensitivity separately. Never clip
negative values or normalize a nonpositive reference gain.

The budget is the smallest measured decoded file meeting an absolute NLL target,
selected on calibration prompts and checked on verification prompts. Targets
0.1, 0.25, 0.5 and 1.0 nats are fixed before scoring. Keep failures and zero-bit
base solutions. Plot file bits and file bits/base parameter count. The latter
does not mean a fraction of physical NF4 model storage. No interpolated files.

## First experiment: matched mismatch, different correction structure

Two frozen NF4 receivers, Mistral-7B and Qwen2.5-7B, one seed, three policies,
three nested training sizes 16/64/256: 18 training runs. Each uses rank-16
all-linear LoRA and one 512-update schedule, with checkpoints 32/128/512.
Every update contains 16 examples. Score complete canonical responses plus EOS
with the existing scorer, which also supplies per-example losses. Prompts are
masked and no response may be dropped or truncated.

Policies use identical prompts and exactly balanced sixteen-label histograms:
the first maps signal A through a shared label code, the second maps A xor B,
and the third changes the second policy's labels on 25% of entities. We do not assume this ordering
implies an ordering in learning difficulty. Entity keys and irrelevant request
IDs do not reveal the policy. Labels for exceptions are fixed across contexts.

The matching-only entity pool is disjoint from all training/evaluation entities.
Before any training, the full-response base-NLL spread across policies must be
at most 0.1 nats/token per model and seed. A failure stops training and requires
an explicit design amendment; no policy is selected using test outcomes. Also
retain action-suffix NLL: predictable common wording can dilute full-text error.
The suffix diagnostic masks the common words within the original response, so
it scores the same action tokens as the full-response measure and includes EOS.

Evaluation keeps a fixed subset of training-eligible entities as n increases,
with fresh request IDs and separate calibration/verification contexts. Retain
covered/uncovered and exception metadata. This is recall under changed context,
not generalization to new random facts. A separate fixed set of unseen entities
tests rule generalization; independent unseen exceptions cannot be predicted.

Reference error is the final predetermined checkpoint, never the best test
checkpoint. At any data size, a change exceeding 5% of the base-to-reference gain
between the last two checkpoints marks optimization unresolved. Passing is only a plateau screen,
not proof of convergence. The action-suffix curve checks whether the learner
acquired the policy rather than only common response wording.

Only largest-n runs receive the initial 5x5 rank/precision compression sweep.
Balanced SVD of the product removes factor gauge, but every result remains an
achieved upper bound within this training route and codec family. Freeze the
base, restore trained factors between interventions, and decode each actual file.

### Amendment 1: label code, before any full training

The first baseline-only check failed the 0.1-nat matching threshold: spreads
were 0.177 on Mistral and 0.137 on Qwen. The simple copy rule started easier.
No pilot training followed. Apply the same fixed bijection
`[7,12,1,14,9,2,15,4,11,0,13,6,3,10,5,8]` to all policy labels, preserving label
counts and exception membership. All policies now learn one common label code;
the simple policy is a one-input lookup and composition still combines inputs.
Recheck on fresh matching entities (`matching_v2`), retaining the original
threshold and all failed-check artifacts. This is calibration, not prospective
evidence of a law. Smokes and baseline checks repeat after the amendment.

Learning-area eligibility also requires the final reference to reach the smallest
registered action-NLL target (0.1); a flat but poor learner must not count as a
valid complexity measurement. Otherwise retain its curve as target-unattained.

## Following experiments, implemented through the same runner

1. `repetition.json`: repeat each training row four times at the same update and
   batch budgets, against a one-copy control. Distinct entity coverage and the
   evaluation universe stay fixed. Checkpoints 128/512/2048 test whether apparent
   sample difficulty is an optimization effect. A repetition effect is an
   optimizer/data-order effect, not added task information.
2. `followups.json`: two fresh seeds, direct training ranks 1/4/16, all three data
   sizes and the longer checkpoint schedule. Compress each trained solution and
   compare at the same absolute target. This tests whether reference rank limits
   learning and whether directly training a small adapter beats recoding a large
   one. Rank-16 alone supplies the primary difficulty predictor; other ranks are
   capacity controls. Do not give each rank its own normalized budget target.
3. After these checks, widen the data range and exception fractions and reserve
   an entire new policy family and receiver for prediction. Compare learning-area
   prediction with base-NLL, corpus-size and constant-budget baselines. Freeze
   model forms before that outer panel. This third stage is a plan, not an
   implemented claim or a submitted GPU campaign; choosing its ranges depends on
   whether the first panel learns and reaches common targets.

Success of the first pilot means the instrument separates initial mismatch,
coverage, optimization and achieved storage with complete artifacts. Six
model-policy points cannot establish a scaling law. No early success licenses
selection of an exponent on the same panel. A negative relation is useful too:
hard-to-find corrections may still have compact implementations.

## Owners and execution

`policy_data.py` owns generation, `policy_complexity.py` reuses ModelSession,
train_adapter, completion_nll and the existing codec; `policy_complexity_analysis.py`
owns finite-range areas and discrete budget selection. Locks include config and
source hashes. Existing experiment data and runners remain separate.

Use `python -m fineqcomp.policy_complexity --config CONFIG --out OUT --phase PHASE
--cell INDEX`. Phases are prepare, base (model index), run (training cell index),
and reduce. Run both smoke model cells first. Then run the pilot base diagnostics
and inspect the mismatch gates before submitting the 18 training cells. A run
lock prevents concurrent duplicate work; completed cells skip on resume. An
interrupted incomplete training cell restarts deterministically.

## Run record: 2026-09-13

The amended real-model smokes completed on Mistral and Qwen (Jean-Zay job
2089199, two cells, exit code zero). The fresh baseline check also completed
(job 2089200). Full-response NLL spreads were 0.01496 for Mistral and 0.01721
for Qwen, so both passed the registered 0.1-nat gate. The 18-cell pilot was then
submitted as job 2098790 with at most two H100 tasks at once. Source and output
locks live under the immutable remote root `fq-policy-99b7297`. That job was
stopped after its first 128-update checkpoints exposed a bad action-only score:
moving the common response prefix into the prompt changed BPE tokens at the new
boundary. Its full-response values remain valid development evidence, but no
cell completed. The corrected score masks within the original response. The
fresh smokes (job 2099103) and matched-base checks (job 2099109) passed from
source root `fq-policy-d5822e1`; corrected pilot job 2099304 then replaced the
cancelled job, again capped at two H100 tasks. The repetition
and rank/seed follow-ups remain unsubmitted pending the pilot result.
