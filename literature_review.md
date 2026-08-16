# Literature review: information limits of quantized fine-tuning

Reviewed on 2026-08-16. This note screens 29 papers at the objective, method,
experiment, and code level. It then studies the six papers that best cover our
research question. Links point to the paper or official project.

## The question we should test

The proposal asks whether the information written by fine-tuning has a measurable
bit budget, whether that budget depends on the data, and how bits trade against
task quality and speed. That contains three separate objects:

1. **Base-model rate:** the bits used to store the frozen model.
2. **Update rate:** the bits used to store the learned change, including scales,
   zero points, shapes, and other decoder data.
3. **Behavioral information:** the bits by which the tuned model shortens the
   code for training behavior relative to a model that did not see that data.

The present campaign mainly varies item 2 after QLoRA training, but its model and
task choices come from work on item 1. It also treats nominal quantizer width as
the rate and task accuracy as the only data-content measure. The closest work
shows that these choices mix distinct questions.

The main study should therefore hold a trained adapter fixed, compress that same
adapter to several exact rates, and measure both task quality and behavioral
information. Random data is useful only as a small capacity calibration. It is
not the main downstream task.

## How the papers were selected

I screened papers on six axes:

- direct match to adapter storage or adapter information;
- match to the “quantized before and after adaptation” condition;
- exact accounting of rate rather than nominal bit width;
- evidence across rates, models, and tasks;
- public code, weights, or enough method detail to reproduce the work;
- feasibility on two 80 GB A100s.

The six finalists are a Pareto set, not the six highest sums. Each supplies a
best-in-pool part that the others lack: direct adapter compression, a bit-level
information measure, a global rate allocator, a mergeable quantized update, a
controlled rate law, or a stable low-bit training method.

## Screened papers

| # | Paper | What it controls | Decision for this project |
|---:|---|---|---|
| 1 | [LoRA](https://arxiv.org/abs/2106.09685) | Low-rank update size and placement | Required baseline, but it does not quantize the base or update. |
| 2 | [QLoRA](https://arxiv.org/abs/2305.14314) | Four-bit frozen base, full-precision LoRA | Required training baseline. Its adapter remains high precision, so it does not answer our storage question. |
| 3 | [GPTQ](https://arxiv.org/abs/2210.17323) | Post-training base-weight quantization | Useful scalar baseline and calibration method; no learned-update rate. |
| 4 | [LoftQ](https://arxiv.org/abs/2310.08659) | Joint quantized-base and LoRA initialization | Strong task recipes and checkpoints; ApiQ gives a stronger low-bit activation target. |
| 5 | [LQ-LoRA](https://arxiv.org/abs/2311.12023) | Low-rank plus quantized base under a global bit budget | **Top six:** exact budget allocation and effective-rate accounting. |
| 6 | [QA-LoRA](https://arxiv.org/abs/2309.14717) | Quantized training base and mergeable quantized result | **Top six:** closest published match to quantized weights before and after tuning. |
| 7 | [ApiQ](https://arxiv.org/abs/2402.05147) | Activation-preserving low-bit base and adapter initialization | **Top six:** strongest reproducible recipe for useful 2–4 bit fine-tuning. |
| 8 | [PEQA](https://arxiv.org/abs/2305.14152) | Learned quantizer scales with fixed low-bit weights | Cheap alternative update parameterization, but it does not measure update information. |
| 9 | [IR-QLoRA](https://arxiv.org/abs/2402.05445) | Information recovery during QLoRA | Better low-bit QLoRA baseline; its “information” is an architecture device, not a codelength. |
| 10 | [QuAILoRA](https://arxiv.org/abs/2410.14713) | Quantization-aware adapter initialization | Relevant initializer; less direct than ApiQ for the final storage question. |
| 11 | [QERA](https://arxiv.org/abs/2410.06040) | Data-aware low-rank reconstruction of quantization error | Good initialization baseline; no compressed learned update or information law. |
| 12 | [PiSSA](https://arxiv.org/abs/2404.02948) | Principal-subspace adapter initialization | Useful raw-adapter control and QPiSSA variant; no exact post-training adapter rate. |
| 13 | [QDyLoRA](https://arxiv.org/abs/2402.10462) | One quantized-base adapter trained for many ranks | Useful rank sweep, but rank is not an exact bit budget. |
| 14 | [AdaLoRA](https://arxiv.org/abs/2303.10512) | Adaptive rank by parameter importance | Useful placement control; full-precision adapter and no rate code. |
| 15 | [DyLoRA](https://arxiv.org/abs/2210.07558) | Nested ranks trained together | Useful efficient ablation; no quantized update. |
| 16 | [LoRAQuant](https://arxiv.org/abs/2510.26690) | Post-training mixed-precision LoRA compression | **Top six:** direct match to our current empirical object. |
| 17 | [How Many Bits Can an Adapter Write?](https://arxiv.org/abs/2607.21351) | Adapter capacity, artifact code, and behavioral information | **Top six:** direct match to the information claim. |
| 18 | [ParetoQ](https://arxiv.org/abs/2502.02631) | Controlled 1–4 bit quantization-aware training laws | **Top six:** best controlled evidence for a bit-width transition and rate law. |
| 19 | [LLM-QAT](https://arxiv.org/abs/2305.17888) | Data-free quantization-aware training of the base | Important low-bit QAT baseline; no adapter or data-compressibility test. |
| 20 | [AWQ](https://arxiv.org/abs/2306.00978) | Activation-aware base-weight quantization | Strong deployment baseline; it does not study adaptation. |
| 21 | [OmniQuant](https://arxiv.org/abs/2308.13137) | Learned weight clipping and equivalent transforms | Strong 2–4 bit base baseline used by ApiQ; no learned-update rate. |
| 22 | [SpQR](https://arxiv.org/abs/2306.03078) | Sparse outlier plus quantized base storage | Exact low-bit storage is useful, but the object is the base model. |
| 23 | [SqueezeLLM](https://arxiv.org/abs/2306.07629) | Dense-and-sparse base quantization | Useful rate-accounting precedent; no adaptation. |
| 24 | [QuIP](https://arxiv.org/abs/2307.13304) | Incoherence processing for low-bit base weights | Strong extreme PTQ baseline; no learned update. |
| 25 | [QuIP#](https://arxiv.org/abs/2402.04396) | Improved incoherence processing and lattice code | Strong 2-bit base result; not an adapter study. |
| 26 | [AQLM](https://arxiv.org/abs/2401.06118) | Additive vector quantization below three bits | Strong base-rate frontier and code; not a fine-tuning information test. |
| 27 | [CALDERA](https://arxiv.org/abs/2405.18886) | Quantized matrix plus quantized low-rank factors | Close algebraic form, but aimed at base compression rather than task updates. |
| 28 | [QR-Adaptor](https://arxiv.org/abs/2505.03802) | Gradient-free downstream rank and bit allocation | Close allocation objective and useful later baseline; less public evidence than the selected methods. |
| 29 | [S-LoRA](https://arxiv.org/abs/2311.03285) | Serving many distinct LoRA adapters | Motivates adapter memory and speed metrics, but does not compress or measure their content. |

## Pareto set: detailed notes

### 1. LoRAQuant: the direct post-training baseline

**Why it is in the set.** This is the closest empirical paper. It trains a LoRA,
then compresses the adapter rather than the frozen model. It evaluates rates
below two bits on tasks where the raw adapter has clear value.

**Method.** For each update `BA`, LoRAQuant computes `U S V^T` and refactors it
as `B' = U sqrt(S)` and `A' = sqrt(S) V^T`. It chooses the smallest leading rank
whose squared singular values explain a fraction `rho` of the total. It stores
that leading part at two or three bits and the rest with a one-bit sign code.
Round-to-nearest quantization uses groups of 128. A short straight-through
optimization reduces the Frobenius error of each paired column and row.

**Models and data.** The paper uses LLaMA-2-7B, LLaMA-2-13B, and Mistral-7B-v0.1.
It trains rank-16 LoRA on all linear layers:

- `meta-math/MetaMathQA`, tested on GSM8K and MATH with exact-answer accuracy;
- `ise-uiuc/Magicoder-Evol-Instruct-110K`, tested on HumanEval pass@1;
- `EdinburghNLP/xsum`, tested with ROUGE-L.

This choice matters. The authors use MetaMathQA rather than the small GSM8K
training split, while one adapter still gets two held-out math measures.

**Training configuration.** The paper reports AdamW with betas `(0.9, 0.95)`,
learning rate `2e-4`, cosine decay, 30% warmup, zero weight decay, FP16,
gradient norm 1, two epochs, and one GPU. Sequence length is 1024 for math and
summarization and 4096 for code. Global batch is 16 for 7B and 8 for 13B. The
released training code sets rank 16, `lora_alpha = 2 * rank`, zero dropout, and
all linear targets, with an NF4 double-quantized frozen base.

**Compression grid.** The useful named points are `2@0.8`, `2@0.9`, `3@0.8`,
and `3@0.9`, where the first value is the leading-part bit width and the second
is the explained-variance threshold. Keep FP16, binary, uniform RTN-2, GPTQ-2,
and a method-specific mixed-precision baseline.

**Evidence.** On Mistral-7B, FP16 adapter scores are 58.83 GSM8K, 19.46 MATH,
45.12 HumanEval, and 31.96 XSum. `2@0.8` obtains 52.08, 16.43, 35.98, and 33.26
at task-dependent rates around 1.82–1.86 bits per adapter value. `3@0.9` obtains
53.75, 18.70, 43.90, and 32.75 at 2.76–2.83 bits. The task dependence of the
realized rate is itself evidence against using only the requested bit width.

**Code audit.** Official code: [MANGA-UOFA/LoRAQuant](https://github.com/MANGA-UOFA/LoRAQuant),
inspected at `d8a5745`. It now includes a PEFT training script, pretrained
adapters, quantizers, and evaluation. The method text says optimization converges
within 100 steps; the main function defaults to 100 but the active high-rank call
uses 300. We should pin our choice and report it rather than inherit it by chance.
The code computes a nominal payload count; our implementation must also serialize
and count scales, zero points, padding, shapes, and headers.

**What to take.** Use these three task families, rank-16 all-linear adapters,
the four mixed-precision points, and post-process one raw checkpoint into every
codec. Add exact file and entropy-coded rates, which the paper does not fully
settle.

### 2. How Many Bits Can an Adapter Write?: the measurement paper

**Why it is in the set.** This paper asks the information question directly. It
separates parameter count, artifact size, and behavior written into a frozen
model. It also shows why random mappings should not be our main task.

**Method.** For a sequence `x`, codelength is `-log2 p(x | theta)`. Unintended
memorization is the amount by which the tuned model shortens the reference code,
clipped so a worse tuned model cannot create negative memorization. The paper
also defines an adapter artifact codelength by quantizing and entropy coding each
tensor and choosing the shortest precision, plus behavioral write bits and the
train-minus-held-out excess.

**Capacity calibration.** Uniform random sequences contain 64 tokens from a
2,048-token alphabet after a start token, hence 704 known bits per item. The
models are 2M- and 8M-parameter GPT decoders pretrained on WikiText-103. Runs use
16,000 fixed steps, no early stopping, and a learning-rate sweep; 40,000 steps
raise the lower-bound estimate by 10–24%. This is a calibration test on small
models, not evidence that a 7B LoRA learns useful random mappings.

**Key findings.** LoRA stores about 1.7–2.8 behavioral bits per trainable
parameter under this protocol, below matched full fine-tuning. At about 37K
trainable parameters, attention-only placement gives 1.30 bits per parameter,
all-module placement 2.14, and MLP-only placement 2.43. Casting trained adapters
to BF16 or FP16 retains almost all measured memorization. The frozen base also
matters: the same adapter memorizes 29% with a random base, 79% with a synthetic
pretrained base, and 98% with a WikiText-pretrained base.

**Real-data audit.** The paper uses Qwen2.5-0.5B-Instruct with rank-4 LoRA on
three-digit arithmetic and key-value tasks, comparing SFT and GRPO at 100–800
steps. It measures train and held-out write bits, canary extraction, and
membership inference. This shows how to use the bit measure on real behavior.

**Code status.** The paper says it releases a measurement harness, but I could
not find a public repository link in the paper, arXiv record, or a title/author
search as of 2026-08-16. The method is specified well enough to implement with
token log probabilities and a real entropy coder, but we should record this as
an independent implementation.

**What to take.** Add behavioral write bits and artifact codelength to the main
real-task runs. Compare attention, MLP, and all-linear adapters at matched encoded
file size, not matched rank. Keep the random-sequence test as one small appendix
calibration on a small model.

### 3. LQ-LoRA: exact global rate allocation

**Why it is in the set.** LQ-LoRA treats the requested model size as a global
constraint and includes low-rank factors and quantizer data in the effective
rate. That is a better rate contract than assigning every tensor the same width.

**Method.** Each pretrained matrix is decomposed as `W ~= Q + L1 L2` by
alternating quantization and randomized SVD. An integer program selects a bit
width, block size, and scale precision for each matrix under a target model
budget. A Fisher-weighted form gives high-error directions more weight.

**Search and configuration.** The allocation search uses first- and second-level
widths from 2, 3, and 4 bits, scale types BF16/FP16/FP32, and block sizes drawn
from 16, 32, 64, and 256 depending on level. Main LLaMA-2 runs use rank 64,
`lora_alpha=16`, zero dropout, and learning rate `2e-5`. Continual language
modeling uses half an epoch of C4 at length 1024. Fisher estimates use 10,000 C4
samples at length 1024. A larger compression run uses two C4 partitions plus
WikiText-2 at length 2048 and then stores the low-rank factors in NF8.

**Models and evaluation.** The paper covers LLaMA-2 7B and 70B, RoBERTa-large,
C4 and WikiText-2 perplexity, OpenAssistant instruction tuning, GLUE, five-shot
MMLU, Vicuna questions, and the Open LLM benchmark set. Target rates range from
2.5 to 4.0 bits. At target 2.75, the full effective rates are 2.95 bits for 7B
and 2.85 for 70B. The paper finds marked loss near 2.5 bits and warns that small
perplexity changes do not predict GSM8K and ARC loss.

**Code audit.** Official code: [HanGuo97/lq-lora](https://github.com/HanGuo97/lq-lora),
inspected at `c2424b3`. It includes packing code, the integer allocator, Fisher
data preparation, training scripts, logs, and 7B/70B checkpoints. Some scripts
retain lab-local paths, but the core allocation and packed-rate code are usable.

**What to take.** Make the encoded adapter budget global, allow per-tensor
allocation, and report requested versus effective rate. Use downstream task
quality, not perplexity alone, to choose a point.

### 4. QA-LoRA: quantized before and after adaptation

**Why it is in the set.** QA-LoRA matches the proposal's state transition most
closely: the base is quantized during training, and the learned change can merge
into a quantized model without a second lossy quantization pass.

**Method.** It groups each input dimension for both quantization and adaptation.
Each group gets its own scale and zero point, while an input pooling operation
reduces the degrees of freedom of the LoRA `A` factor. The learned low-rank term
can then update the group zero points and merge into the low-bit base.

**Configuration.** Main experiments use asymmetric GPTQ at 2, 3, and 4 bits,
group size 32, `act_order=false`, and `true_sequential=true`. Models are LLaMA
7B/13B/33B/65B and LLaMA-2 7B/13B. Training uses Alpaca 52K or a 320K sample of
FLAN v2. It uses paged AdamW, max gradient norm 0.3, batch 16, a constant
learning rate of `2e-5` for 7B/13B and `1e-5` for 33B/65B, 10K Alpaca steps or
20K FLAN steps. The paper reports that low-bit models need more data and that
320K FLAN items suffice for its INT2 and INT4 curves.

**Evaluation.** The main measures are zero- and five-shot MMLU plus BoolQ, PIQA,
SocialIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, and OpenBookQA. This is
a good general-skill check, but the instruction data and MMLU score do not isolate
the amount of task-specific information written.

**Code audit.** Official code: [yuhuixu1993/qa-lora](https://github.com/yuhuixu1993/qa-lora),
inspected at `91604c7`. It has the modified PEFT layer and merge operation. It
depends on Python 3.8 and AutoGPTQ 0.3.0 and asks users to replace an installed
library file. Its README notes a conflict with current AutoGPTQ. Reuse the math,
not the old environment or file-replacement setup.

**What to take.** Add one mergeable-quantized-update baseline and test the final
merged artifact, not just a dequantized side adapter. Use group 32 as the paper
point. Do not use QA-LoRA as the sole method because its constrained update is a
different writable object from a normal LoRA.

### 5. ParetoQ: the controlled bit-law reference

**Why it is in the set.** ParetoQ is the strongest direct precedent for a common
training and evaluation protocol across bit widths and for fitting empirical
rate laws. It is about the whole model, not the adapter, so it guides analysis
rather than the main training recipe.

**Method and grid.** It uses one quantization-aware training framework across
binary, ternary, 2-, 3-, and 4-bit weights. It quantizes all weights except
embeddings and output layers and starts from pretrained weights. Models span
MobileLLM 125M–1.5B and LLaMA-3 1B, 3B, and 8B. Evaluation uses WikiText-2
perplexity and eight zero-shot tasks: ARC-Easy, ARC-Challenge, BoolQ, PIQA,
SocialIQA, HellaSwag, OpenBookQA, and WinoGrande.

**Training configuration.** The main low-bit runs use AdamW with no weight decay,
16 GPUs, and batch 8 per GPU. Binary, ternary, and 2-bit runs use 120K steps at
`2e-5`; 3- and 4-bit runs use 40K steps at `1e-5`; both use cosine decay to zero.
The paper finds that 3- and 4-bit fine-tuning tends to saturate around 10B tokens,
while 1-, 1.58-, and 2-bit runs need about 30B. It describes a change between
2 and 3 bits: higher widths can adjust within a nearby quantization grid, while
lower widths must rebuild representations.

**Code audit.** The original official repository has moved into
[torchao/prototype/paretoq](https://github.com/pytorch/ao/tree/main/torchao/prototype/paretoq),
inspected at `d37142a`. The public Hugging Face example uses Llama-3.2-1B,
sequence length 2048, BF16, batch 2, one epoch, `2e-5`, and cosine decay. It is a
small reproduction, not the paper-scale run. The shell script claims to accept a
bit argument but currently hard-codes `--w_bits 4`; fix that before reuse.

**What to take.** Treat the 2-to-3-bit region as a pre-registered test, fit the
same rate curve to all methods, and give very-low-bit points enough optimization.
Do not copy the 10–30B-token budget to adapter training; it applies to whole-model
quantization-aware training and is not feasible for our main study.

### 6. ApiQ: the practical low-bit training control

**Why it is in the set.** ApiQ gives the best detailed and released recipe for
keeping a 2–4 bit frozen model useful during task fine-tuning. It also tests the
same Mistral-7B model and math tasks that fit our plan.

**Method.** ApiQ jointly initializes the quantized base and LoRA by minimizing
activation error between the full-precision and quantized networks. The block-wise
form processes transformer blocks in order so errors from earlier blocks enter
the next calibration step. It then freezes the low-bit base and fine-tunes the
BF16 adapter.

**Quantization configuration.** It samples 128 calibration sentences, uses LoRA
on all linear layers, rank 64, and uniform affine weights. Group 64 is the default
and the paper also tests group 128. The block-wise form takes about one hour to
calibrate LLaMA-2-7B in the reported setup, versus four hours for the layer-wise
form.

**Tasks and training.** The useful 7B experiments cover LLaMA-2-7B,
LLaMA-2-13B, and Mistral-7B-v0.1 at 2, 3, and 4 bits:

- GSM8K: six epochs, batch 16, max length 512, cosine schedule, 3% warmup,
  weight decay 0.1, and a per-model learning-rate search. Mistral uses `7e-5` in
  the reported best settings. Results are means and standard deviations of
  three runs after selecting the learning rate.
- Math10K: three epochs, batch 16, max length 512, `3e-4`, linear schedule, 10%
  warmup, tested on GSM8K, SVAMP, MAWPS, and AQuA.
- A combined 170K commonsense set: the same three-epoch multi-task recipe,
  tested on eight common reasoning benchmarks.
- WikiText-2: three epochs, batch 64, length 1024, with perplexity evaluation.

At Mistral-7B, block-wise ApiQ reports GSM8K 59.2, 56.0, and 38.3 at 4, 3, and
2 bits. QLoRA and LoftQ collapse at the reported 2-bit point. This makes ApiQ a
useful control for whether a weak low-bit result comes from adapter capacity or
from a damaged frozen base.

**Code audit.** Official code: [BaohaoLiao/ApiQ](https://github.com/BaohaoLiao/ApiQ),
inspected at `bb9e9a6`. It includes calibration and task scripts. The GSM8K script
matches six epochs, learning rate `3e-4`, microbatch 4, accumulation 2, weight
decay 0.1, 3% warmup, and cosine decay. The repository warns that only real
symmetric checkpoints work in its fine-tuning path because of an AutoGPTQ issue,
while the paper's main checkpoints are asymmetric. Treat this as a known
reproduction risk.

**What to take.** Use block-wise activation preservation as a low-bit-base
control, all-linear placement, three paired seeds, and multi-task math as a way
to obtain several held-out measures from one trained adapter.

## What the literature changes in our experiment design

### 1. Separate the two rate questions

The primary paper should study **adapter storage rate** with one frozen-base
format. A smaller secondary study can vary the base format using QLoRA, ApiQ, or
QA-LoRA. Crossing five base widths with five adapter widths would make the main
result hard to identify and would spend most of the run on a different question.

### 2. Train a useful adapter once, then derive every rate point

For each model, task, and seed:

1. Train one FP16/BF16 rank-16 all-linear adapter on a four-bit NF4 base.
2. Verify a held-out gain before starting the full codec evaluation.
3. Serialize FP16, uniform 8/4/3/2-bit, binary, and the four LoRAQuant points
   from the same adapter.
4. Evaluate every artifact with the same prompts and generation settings.

This changes codec evaluation from dozens of training runs into paired
post-processing. The pair is important: each rate point shares the same learned
update, so training variance cannot look like a compression effect.

### 3. Use tasks with known learning signal

Use base rather than instruction-tuned checkpoints for the main task-learning
study.

| Role | Train data | Held-out measures | Why |
|---|---|---|---|
| Math | `meta-math/MetaMathQA` | GSM8K exact match and MATH exact match | Direct LoRAQuant replication; large training set; two tests per adapter. |
| Code | `ise-uiuc/Magicoder-Evol-Instruct-110K` | HumanEval pass@1 | Structured generation that is sensitive to lost update detail. |
| Summarization | `EdinburghNLP/xsum` | ROUGE-L, plus validation NLL | Different output entropy and a dense metric. |

The exact replication model should be `mistralai/Mistral-7B-v0.1`. The modern
extension should be `Qwen/Qwen2.5-7B`. Qwen2.5 has an open base checkpoint and
connects to the real-data information audit in *How Many Bits Can an Adapter
Write?*. LLaMA-2-7B can replace Qwen for a strict two-model replication, but it
is gated and gives less new evidence.

Before the three-seed campaign, train one raw adapter for each model-task pair.
Require a material held-out gain over the base, expressed in the task's natural
metric. If one pair fails, change that pair's training recipe before producing
any compressed points. Do not pick a task based on base difficulty alone: the
gate tests actual learnability.

### 4. Measure rate and information, not just width

Record all of the following for each artifact:

- nominal payload width;
- total packed weight bits;
- scale, zero-point, padding, shape, and header bits;
- complete file bytes;
- entropy-coded adapter codelength;
- GPU-resident bytes after loading;
- task score and retained gain
  `(compressed - base) / (raw_adapter - base)`;
- train and held-out negative log likelihood in bits;
- behavioral write bits and train-minus-held-out excess.

The retained-gain ratio solves a problem in the old task grid: a compressed
adapter should get credit only for preserving what fine-tuning added, not for
knowledge already present in the base.

### 5. Match placement comparisons by encoded budget

Compare attention-only, MLP-only, and all-linear adapters at the same final file
budget. Choose ranks after counting tensor shapes and codec overhead. A fixed
rank comparison changes both placement and storage, and the adapter-capacity
paper shows that equal parameter counts still do not give equal writable bits.

### 6. Keep broad skill evaluation small

Run MMLU or the eight-task reasoning set on only four points per raw adapter:
base, raw adapter, best sub-two-bit artifact, and best near-three-bit artifact.
Do not run IFEval for every codec. The prior campaign showed that full IFEval can
cost more time than the task experiment. It also measures instruction following,
which is not a clean control for base checkpoints trained on math, code, or XSum.

### 7. Do not claim speed from a packed file alone

If the loader expands the adapter to FP16, compression improves disk and transfer
cost but not resident memory or matrix multiplication. Report separately:

- encoded file size and load time;
- peak and steady GPU memory;
- warm tokens per second at fixed batch and sequence length;
- adapter-switch time in a multi-adapter setting.

Only call a point faster if a packed inference kernel actually uses the low-bit
factors.

## Recommended first complete campaign

### Training runs

- Models: Mistral-7B-v0.1 and Qwen2.5-7B base.
- Tasks: MetaMathQA, Magicoder-Evol-Instruct-110K, and XSum.
- Seeds: 11, 22, 33.
- Adapter: rank 16, alpha 32, zero dropout, all linear layers.
- Frozen base: NF4 with double quantization.
- Optimizer: AdamW `(0.9, 0.95)`, learning rate `2e-4`, cosine decay, 3% warmup
  in the released code or 30% in the paper appendix, zero weight decay, gradient
  norm 1, two epochs.
- Length: 1024 for math/XSum and 4096 for code.

The warmup mismatch must be resolved in a raw-adapter pilot. Use one value for
all final runs and record why. This gives 18 raw training runs, not one training
run per codec.

### Artifacts derived from each raw adapter

- FP16 reference;
- uniform scalar 8, 4, 3, and 2 bit;
- one-bit sign baseline;
- LoRAQuant `2@0.8`, `2@0.9`, `3@0.8`, and `3@0.9`;
- optional global-budget allocation at target effective rates 1.5, 2.0, 2.5,
  3.0, and 4.0 bits if the allocator is ready before launch.

The scalar and LoRAQuant artifacts must use the same group size, rate accounting,
and serialized container. Otherwise method labels also change the denominator.

### Secondary controls

- Mistral-7B only: NF4 QLoRA versus 2/3/4-bit ApiQ block-wise initialization on
  the math adapter. This identifies damage from the base quantizer.
- Mistral-7B math only: attention, MLP, and all-linear placements at matched
  encoded adapter budgets.
- Small GPT model only: the random 704-bit sequence capacity calibration. Do not
  use random mappings for the downstream claims.
- One model and task: QA-LoRA merged 2/3/4-bit artifacts to test whether a final
  quantized model changes the result relative to a side adapter.

### Main figures

1. **Adapter rate-distortion:** retained task gain against complete encoded
   adapter bits, with paired seeds and one panel per task family.
2. **The 2-to-3-bit region:** a close view with raw points and a shared segmented
   or monotone fit; do not assert a threshold unless its location repeats across
   models and seeds.
3. **Artifact bits versus behavior bits:** entropy-coded adapter codelength on the
   x-axis and train/held-out behavioral write bits on the y-axis.
4. **Matched placement:** attention, MLP, and all-linear retained gain against the
   same encoded budgets.
5. **Allocation map:** effective width by layer, module type, and singular
   direction for the best mixed-precision method.
6. **System Pareto plot:** task score against file bytes, GPU-resident bytes, load
   time, and measured throughput. Use separate panels when one codec expands on
   load.

### Claim gates

- **Useful adaptation:** the raw adapter gives a material held-out gain on every
  reported model-task pair.
- **Rate claim:** all metadata and padding are included in the rate; requested
  and effective rates are both reported.
- **Compression claim:** at least one sub-three-bit method retains at least 90%
  of raw-adapter gain in two task families and both models.
- **Information claim:** behavioral write bits change with task/data condition
  after controlling for encoded adapter size; file size alone does not meet it.
- **Threshold claim:** the fitted transition repeats across both models and at
  least two task families with seed uncertainty shown.
- **Speed claim:** the packed representation remains packed during measured
  inference and improves a timed system measure.

## Bottom line

The strongest main question is not “does a 3-bit LoRA still work?” LoRAQuant has
largely answered that. The stronger question is:

> At a fixed encoded update budget, how much task-relevant behavior can an
> adapter write, and how do data, placement, and the frozen quantized base change
> that rate-distortion curve?

That question joins the direct adapter result from LoRAQuant, the information
measure from *How Many Bits Can an Adapter Write?*, and the controlled rate logic
from LQ-LoRA and ParetoQ. It also gives clear negative results: no behavioral
dependence means the proposal's data-compressibility claim fails even if 3-bit
adapters preserve accuracy.
