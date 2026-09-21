# fineQComp: what would make a strong paper?

The [experiment ledger](ledger.md) comes first. The current evidence establishes neither a universal information law nor calibrated early prediction. It does establish that the old codec obscured the target, that receiver/task interactions matter, and that the finished checkpoint can contain severe behavioral damage that compression removes.

My first choice is **F01: an explicit test of the link between learned predictive information and deployable correction rate**. Its distinctive result must be a prospective quantitative law with a stated domain, or a controlled separation disproving that law. F02 is the next choice if a short optimization prefix is necessary. F03 is the strongest route from the existing positive artifacts to a paper, provided the recovery survives simple shrinkage controls.

## Define the object before fitting a law

Let (M\) be the exact frozen receiver, (Q\) a distribution of task inputs, (f\) the desired behavior, and (\mathcal C\) a public menu of prefix-free messages and decoders. The menu fixes software, shared dictionaries, numerical precision, side information, training allowed at the receiver, and inference compute. Define

\[
B_{\mathcal C}(d;f,M,Q)=\min_{c\in\mathcal C}\{\ell(c):\mathbb E_Q[d(f(X),\operatorname{Dec}(M,c)(X))]\le d\}.
\]

Here (\ell(c)\) is the **actual transmitted bit count**, including code identifiers, seeds, scale parameters, indexes and nonpublic dictionaries. Shared decoder software must be fixed before the task draw. Report its one-time cost separately and its amortization explicitly. A task-specific decoder is part of the message. Access to an external executor, retrieval table or extra training is part of the resource contract, not free information.

There are three different targets:

- **Task rate:** achieve a fixed externally defined utility, or task distortion (d\). Two implementations with equivalent behavior qualify equally.
- **Imitation rate:** approximate the behavior of the finished checkpoint (f_T\). This may preserve mistakes or formatting changes.
- **The current gain-retention rate:** satisfy (L(M)-L(c)\ge .9[L(M)-L(f_T)]\). This is a useful operational comparison, but a scalar utility constraint does not preserve the function or the whole conditional distribution.

The minimum over a tested menu is an **achievable upper bound**, not the unrestricted minimum description length. The audit's interpolated crossing is neither a file nor a certified lower bound. Keep actual passing files, adjacent failing files, uncertainty and censoring. Functional distortion makes the *qualification* invariant to parameterization; it does not magically make the achievable rates of restricted code families equal.

For a random task/function (F\) drawn from a declared family, conditional Shannon rate–distortion is well-defined:

\[
R_{F\mid M}(d)=\inf_{P(\hat F\mid F,M):\,\mathbb E d(F,\hat F)\le d} I(F;\hat F\mid M).
\]

This is an ensemble quantity. It is not a mutual information obtained from one adapter without specifying a random experiment. Under the usual coding assumptions it lower-bounds expected message length; it does not assert that SGD plus LoRA attains that bound.

For (K\) independent fair binary correction components absent from the receiver, and average Hamming distortion measured on those components, the asymptotic benchmark is (K[1-h_2(d)]\) bits. There may also be a fixed implementation cost. This suggests an additive overhead plus an information-dependent term, not an automatic power law in dataset rows and model parameters. With shared modules, nonuniform task queries, incomplete learning or a restricted decoder, the form changes.

In (B^*\approx C I^\alpha N^\beta M^\gamma\), (M\) cannot ambiguously mean both a receiver and its parameter count. Use (P\) for parameter count; (N\) for distinct independently informative training units; (T\) for optimization exposure; and (I\) only for an explicitly defined information quantity. A twofold increase in (P\) need not change conditional correction information monotonically. A receiver-independent power law is a hypothesis, not a definition.

## Ranking

Scores are **novelty / expected signal / feasibility / defensibility / paper upside**, each 1–5; higher is better. They assess the specific designs below, not the broad topic. Order reflects decision value for this project, rather than an unweighted sum. Compute is incremental pilot / broader confirmation in aggregate H100-hours; overlapping directions reuse work. All models below are proposals whose revisions must be pinned. Every old Qwen/Mistral/Llama/Gemma panel belongs to discovery. Reserve an unused family, provisionally [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct), for a sealed final test after a competence-only qualification.

| Rank | Direction | Scores | H100-hours |
|---:|---|---|---:|
| 1 | F01 — Storage–learning duality, tested prospectively | 4 / 3 / 3 / 5 / 5 | 60–120 / 300–900 |
| 2 | F02 — When does the final budget become predictable? | 4 / 4 / 4 / 5 / 5 | 40–100 / 200–600 |
| 3 | F03 — Useful corrections and damage have different rate frontiers | 4 / 5 / 5 / 4 / 4 | 10–35 / 80–240 |
| 4 | F04 — Causally change what the receiver already knows | 4 / 4 / 4 / 5 / 4 | 30–80 / 180–500 |
| 5 | F12 — Shared corrections create subadditive storage | 4 / 4 / 4 / 4 / 4 | 30–70 / 160–450 |
| 6 | F13 — Average distortion hides expensive rare corrections | 3 / 5 / 5 / 5 / 4 | 10–30 / 80–200 |
| 7 | F06 — A replay code competes with a parameter code | 3 / 4 / 4 / 5 / 4 | 20–60 / 120–350 |
| 8 | F10 — Information storage and rule discovery separate | 3 / 4 / 4 / 5 / 4 | 30–80 / 150–400 |
| 9 | F11 — Program identity versus parameter payload | 3 / 3 / 4 / 5 / 4 | 25–70 / 120–350 |
| 10 | F16 — High risk: the missing variable is execution cost | 5 / 3 / 2 / 4 / 5 | 60–150 / 400–1,200 |
| 11 | F07 — How much rate remains after quotienting implementations? | 3 / 4 / 4 / 4 / 3 | 20–60 / 100–300 |
| 12 | F14 — A representability transition, not a rank sweep | 3 / 4 / 4 / 5 / 3 | 25–70 / 120–350 |
| 13 | F05 — A functional distortion profile replaces one complexity scalar | 3 / 4 / 4 / 4 / 4 | 25–60 / 150–400 |
| 14 | F08 — Translation cost between receivers | 3 / 3 / 3 / 4 / 4 | 40–100 / 200–600 |
| 15 | F09 — Test whether optimization-path information is real or arbitrary | 3 / 3 / 4 / 5 / 3 | 20–60 / 100–300 |
| 16 | F15 — Singular learning theory must predict code lengths, not correlations | 2 / 2 / 2 / 4 / 3 | 40–120 / 250–700 |

## Sixteen distinct directions

### F01 — Storage–learning duality, tested prospectively

**Abstract claim.** Under a stated learner and decoding contract, an early predictive-information curve forecasts the bytes needed to deliver a specified correction on unseen tasks and receivers, and identifies where that relation breaks.

**Closest repo work / difference.** The known-payload prequential study measured learner-extractable structure, while the eight-update probe prices one update direction. This tests a communication-theoretic relation between a *learning curve* and a *separate deployable code*, rather than renaming another gradient statistic “information.”

**Definitions.** Record prequential label codelength (L_{\rm pre}(n)\) with a fixed online learner and fresh blocks; subtract (n\) times independently estimated final population loss to obtain the finite-sample excess description length. It is learner-dependent and need not be nonnegative for an arbitrary damaging training path. The forecast uses only prefix observations and a frozen extrapolation rule; the final loss is never an input to an early forecast. Target the task rate above, with imitation and old gain-retention rates as diagnostics.

**Experiment.** Small transformers on independent correction modules with (K=4,16,64,256,1024,4096\), different known receiver overlap, and both shared-rule and unstructured families. Then Qwen2.5-7B/Mistral-7B development on verified schema mappings, finite-field programs and executable code edits; seal Olmo and two new generator families. Vary source complexity independently of rows, repetition, noise and updates. Compare fixed replay, balanced-SVD adapter and a fixed compact correction-module code; charge all metadata. Reuse existing early checkpoints and run reducers.

**Signature / falsifier.** One frozen predictor beats the development-trained corpus mean in prospective log-byte error, has small bias and useful intervals, and predicts an actually passing code. Failure of calibration, or code-family-dependent order reversals after matching utility, rejects the claimed duality.

**Useful failure.** A controlled gap between predictive learning information and implementation bits would show exactly why “bits learned” does not price a deliverable correction. It is publishable only with a mechanism and a bound, not another negative scatterplot.

**Novelty.** [Excess Description Length (2026)](https://arxiv.org/abs/2601.04728) already defines the learning-side quantity; [conditional task complexity (ICML 2026)](https://arxiv.org/abs/2602.15829) already compares adaptation descriptions. The new result must be the quantitative bridge—or an explicit separation—with sealed forecasts. Merely implementing either paper is insufficient.

**Theory / resources.** Prove a duality for a Bayesian finite correction family and bound excess implementation rate for a restricted decoder. Then exhibit a learner for which it fails. 60–120 pilot / 300–900 confirmation hours. Reuse prequential instrumentation, `on_update`, new rank frontiers and existing model caches. **Scores: 4/3/3/5/5.**

### F02 — When does the final budget become predictable?

**Abstract claim.** Final correction rate becomes forecastable only after a measurable amount of optimization evidence reveals which behavior the learner will acquire; the observation threshold transfers across receivers.

**Closest repo work / difference.** One/eight-update pricing and the wider sketch already failed calibration. This studies the *information available to a forecaster*, separating uncertainty about the endpoint from poor scaling of a chosen sketch. It does not simply try 16, 32 and 64 updates until a correlation improves.

**Definitions.** Let (O_k\) contain a preregistered set of observations up to update (k\): checkpoint code curves on calibration data, online losses and task-level functional predictions. Define forecast risk for (Y=\log_2 B\), and conditional endpoint spread under independently continued training randomness. Define (k_\epsilon\) as the first locked observation budget attaining an error/coverage target. Count data inspected and GPU time as well as updates.

**Experiment.** Fork existing prefix checkpoints into different future data orders/seeds; create tasks with early-identical evidence but later revealed sharing rules. Compare an early function forecast, an early byte-curve forecast, corpus means and a full-information retrospective oracle. Qwen/Mistral development; sealed new tasks and Olmo. Fix total training exposure and use acquisition gates. Forecast the distribution of attainable file budgets, including censoring, before evaluating endpoint files.

**Signature / falsifier.** A sharp reduction in prospective risk coincides with functional branch selection and holds at new horizons/families without an intercept. If a cheap (k=0\) task descriptor matches the best later forecast, the optimization-observation claim fails. If endpoint spread stays large at all cheap prefixes, the early-prediction objective fails in a measurable way.

**Useful failure.** Establish a practical limit on early-budget forecasting and report which observation, rather than which arbitrary scalar, removes it. A failure of one regressor is not an impossibility result.

**Novelty.** [EDL](https://arxiv.org/abs/2601.04728) studies information gained during learning, while [grokking complexity dynamics](https://arxiv.org/abs/2412.09810) studies changing compressibility. Neither alone supplies an observation-budget forecast test for compressed corrections. The proposed lower bound must explicitly restrict observation/computation access.

**Theory / resources.** Use a two-world indistinguishability bound and an exact Bayes-risk decomposition; see the full design below. 40–100 / 200–600 hours, largely reusing budget probes and saved trajectory checkpoints. **Scores: 4/4/4/5/5.**

### F03 — Useful corrections and damage have different rate frontiers

**Abstract claim.** The rate required to retain task improvement differs systematically from the rate required to imitate a finished fine-tune, because training can add compact useful corrections and removable harmful behavior.

**Closest repo work / difference.** Spectral/scale recovery already finds large Qwen recovery and a strong .3-scale control. The new claim is a causal *two-objective frontier* across acquisition and damage, not that binary quantization has found a special semantic direction.

**Definitions.** On an independent base evaluation, separate problems where the receiver is correct from those it misses; estimate acquired successes and newly damaged successes on fresh paired samples. Report their two-dimensional frontier versus bytes, plus full-checkpoint imitation loss. Never define a “useful component” by selecting favorable test examples after compression.

**Experiment.** Cross aligned/permuted supervision, base headroom and corruption prevalence while holding content and exposure fixed. Evaluate calibrated rank/precision, scalar shrinkage, rank-1 BF16, norm-matched and random-direction controls. Reuse all three-seed Qwen recovery files and Mistral trajectories; confirm on untouched task templates and a held-out family. Verify complete code outputs and use GSM-Symbolic as development evidence, not a new prospective test.

**Signature / falsifier.** A code preserves acquisition while removing damage, and its task-rate ordering differs from imitation rate on sealed examples. If one scalar shrinkage explains the full frontier across both partitions, the proposed selective mechanism fails; report shrinkage as the result. If improvement only restores base behavior, “stored new skill” is false.

**Useful failure.** A calibrated counterexample to likelihood-retention as a measure of useful learned behavior remains valuable, even without a new compressor. A generic post-hoc recovery trick is less valuable.

**Novelty.** [LoRA Learns Less and Forgets Less (2024)](https://arxiv.org/abs/2405.09673) already studies learning/forgetting; [LoRA-Squeeze (2026)](https://arxiv.org/abs/2602.10993) already favors training wide then compressing. Distinguish this through measured task-versus-imitation rates and controlled harmful supervision, with shrinkage given full credit.

**Theory / resources.** Construct orthogonal useful/harmful function components showing no ordering between imitation rate and useful-task rate. 10–35 / 80–240 hours after reusing ongoing recovery/trajectory evaluations. **Scores: 4/5/5/4/4.**

### F04 — Causally change what the receiver already knows

**Abstract claim.** Changing receiver knowledge while holding architecture, target function and task difficulty fixed changes the required correction rate in proportion to the information truly missing from that receiver.

**Closest repo work / difference.** Transposed receivers compare naturally different models and obtain failed prediction; high-gain tasks change both headroom and target. This intervenes on receiver overlap itself.

**Definitions.** Draw a shared function from (K\) independent modules. Install a randomized subset in matched small pretrained receivers; keep a gate hiding or revealing access separate from the stored payload. Missing information is conditional entropy under this known construction, not base NLL. Measure both public-code task rate and restricted adapter rate.

**Experiment.** Tiny controlled transformers: 0/25/50/75/100% overlap, (K\) over three orders of magnitude, balanced exposure and identical architecture. Match base output accuracy using independently randomized access gates, then adapt. Transfer the intervention to Qwen/Mistral through matched preparatory training on disjoint namespaces, and reserve a new family. Include receivers storing irrelevant modules and receivers with the right format but wrong content.

**Signature / falsifier.** Rate follows missing content after matching zero-shot accuracy; revealing latent access costs less than installing payload. A null under verified installed knowledge rejects overlap as the controlling variable for the chosen decoder.

**Useful failure.** It separates a knowledge-information barrier from a receiver's access/implementation barrier; this can overturn the supervisor's simple correction-information account.

**Novelty.** [Physics of Language Models 3.3 (2024/ICLR 2025)](https://arxiv.org/abs/2404.05405) measures storage capacity, and [conditional task complexity](https://arxiv.org/abs/2602.15829) frames adaptation relative to a pretrained model. The new leverage is a randomized knowledge-overlap intervention with matched apparent headroom, not another receiver panel.

**Theory / resources.** Conditional rate–distortion for partially revealed components, plus a gate/payload counterexample. 30–80 / 180–500 hours. Reuse known-payload generators, model loader, train loop and rank codec. **Scores: 4/4/4/5/4.**

### F05 — A functional distortion profile replaces one complexity scalar

**Abstract claim.** Correction storage depends on the distribution of independently necessary functional distinctions, so tasks with equal total information can have predictably different rate–distortion curves.

**Closest repo work / difference.** The repo's spectral/Fisher/log-volume search summarizes parameter or local correction geometry. Here the components are experimentally controlled *input-output distinctions*, and their query frequencies define distortion before any adapter exists.

**Definitions.** For independent fair bits with query weights (w_j\), use (R(d)=\min_{d_j}\sum_j[1-h_2(d_j)]\), subject to (\sum_j w_jd_j\le d\), (0\le d_j\le1/2\). Shared or dependent components require a different source model. A natural-task analogue uses a locked collection of verified correction requirements; it is only an empirical operational profile unless independence is established.

**Experiment.** Hold (K\), total entropy and training count fixed; vary uniform, heavy-tailed and block-correlated query weights. Measure the complete achievable distortion curve, not a fitted r90 scalar. Test several margins (d\), receiver overlaps and rank ceilings. Develop on small models/Qwen; transfer to Mistral and a reserved family on schema/execution tasks.

**Signature / falsifier.** Preregistered water-filling allocation predicts bends and task-order reversals. If measured curves depend mainly on architecture even after acquisition and code controls, the source-distinction profile is insufficient.

**Useful failure.** Quantify the gap between a Shannon envelope and neural implementation, instead of claiming an effective-rank correlation. Natural tasks failing the component decomposition should be reported as such.

**Novelty.** [Fundamental Limits of Prompt Compression (2024)](https://arxiv.org/abs/2407.15504) already applies rate–distortion with fixed receivers and query distributions. This direction concerns learned correction messages and tests predicted *curve shape* against decoded adapters; the information-theory construction itself is established.

**Theory / resources.** Derive finite-block upper/lower bounds for weighted component coding. 25–60 / 150–400 hours; reuse rank frontiers and known-source generation. **Scores: 3/4/4/4/4.**

### F06 — A replay code competes with a parameter code

**Abstract claim.** The shortest way to transmit a learned correction can be a synchronized learning transcript rather than its adapter, and the gap identifies the receiver's implementation overhead.

**Closest repo work / difference.** Earlier prequential experiments estimated label information; they did not compare an actually decoded replay message against actual adapter files at the same utility and total resource contract.

**Definitions.** A replay message contains a losslessly encoded training subsequence, order, random seed and deviations from a public learner. The receiver runs that learner from (M\). Count training inputs unless they are genuinely public side information. Separate one-time decoder compute from recurring inference cost. No “seed-only” coding when the decoder already knows the task's sampled payload.

**Experiment.** Reconstruct completed simple adapters exactly where deterministic replay permits, otherwise reconstruct functionally to a locked tolerance. Compare replay, coreset replay, full adapter and informed rank code on known programs, schema maps and natural code edits. Qwen/Mistral development, untouched Olmo task draws. Vary repetitions, informative samples and decoder training budget.

**Signature / falsifier.** Replay beats parameter transmission by a predictable factor while the decoded model passes sealed utility checks. If nondeterminism, input transmission or decoder costs erase the benefit, reject the proposed cheap code.

**Useful failure.** It provides an auditable upper bound and explains why a small source program can require a large *fast* adapter. It does not turn prequential loss into parameter information by assertion.

**Novelty.** [Conditional task complexity](https://arxiv.org/abs/2602.15829) already counts data and adaptation programs; [EDL](https://arxiv.org/abs/2601.04728) supplies the predictive-code lens. Novelty requires measured replay/parameter crossover and its compute dependence. Without that, this is a control within F01, not a standalone paper.

**Theory / resources.** A constructive two-part upper bound (B\le L(D_{\rm replay})+L(A,\xi)\) under functional replay, with decoder-time constraints explicit. 20–60 / 120–350 hours. Reuse the existing trainer and datasets; avoid a new training engine. **Scores: 3/4/4/5/4.**

### F07 — How much rate remains after quotienting implementations?

**Abstract claim.** Most variation in compressed adapter size across rank and seed can be attributed to redundant implementations of the same correction, and a fixed functional recoding removes that variation.

**Closest repo work / difference.** Balanced SVD removes LoRA factor scaling and rank redundancy for a given update. It does not identify different checkpoints that implement the same function on a declared test distribution.

**Definitions.** Exact equivalence is equal outputs on an exhaustive finite domain. Approximate natural equivalence uses a specified functional metric and held-out confidence bound; “within epsilon” is not a transitive equivalence relation. Use covers or a common representative, not arbitrary connected components. Charge the representative's code and any mapping.

**Experiment.** Train the same learned function at ranks 2/8/32 and three seeds; include exact LoRA gauge transformations as sanity checks. Distill to a fixed existing low-rank student using calibration queries, then compare actual bytes at held-out fidelity. Qwen/Mistral and a sealed family; random-label and unlearned controls distinguish shared function from shared failure.

**Signature / falsifier.** Recoded rates converge substantially more than original file sizes, with equivalence preserved on new queries. If seed differences remain after strong functional matching, the tested recoder cannot remove implementation dependence.

**Useful failure.** A lower bound on practical recoding overhead is useful; failure to find a short code is not proof that none exists.

**Novelty.** [Free Bits from Rotational Symmetries (2024)](https://arxiv.org/abs/2410.01309) already exploits numerical symmetries; [LoRA-Squeeze](https://arxiv.org/abs/2602.10993) already compresses trained rank. Cross-checkpoint *functional* recoding is the added question, not another gauge normalization.

**Theory / resources.** For a finite function class, compare covering numbers with implementation-code multiplicities. 20–60 / 100–300 hours; reuse saved adapters and PEFT distillation paths where present. **Scores: 3/4/4/4/3.**

### F08 — Translation cost between receivers

**Abstract claim.** Correction messages have a portable content part and a receiver-specific implementation part, measurable through asymmetric translation costs between frozen models.

**Closest repo work / difference.** The transposed-receiver panel rescored task budgets across bases. It did not transmit one learned correction to a second receiver under a fixed translation contract.

**Definitions.** (B_{A\to B}(d)\) is the smallest message plus task-specific translator that makes receiver (B\) implement a correction specified through receiver (A\). A shared translator trained on development tasks is allowed but its cost and amortization are reported. Distinguish direct function examples from weight-space maps.

**Experiment.** Learn a generic translator on many small, disjoint symbolic corrections; test novel corrections in both directions Qwen↔Mistral and to held-out Olmo. Compare direct tuning, transmitted examples, compressed adapter and a simple output-level correction table. Vary target complexity, overlap and decoder compute; no test-task translator fitting is free.

**Signature / falsifier.** Translation costs are asymmetric and predictable from verified receiver access, while portable codes preserve new functional tests. If a translator mostly relearns the target from many examples, there is no portable information result.

**Useful failure.** Establish that a behaviorally small correction is not a transferable module, narrowing the meaning of receiver-relative information.

**Novelty.** [GraftLLM / Modular SkillPacks (2025)](https://arxiv.org/abs/2505.18502) already transfers compressed skills. [Usable-information similarity (2026)](https://arxiv.org/abs/2601.21568) treats asymmetric functional stitching. Novelty would be a charged communication rate and held-out translation-cost law, not merely successful transfer.

**Theory / resources.** A conditional coding inequality holds when a translator can simulate the source decoder; identify when its cost exceeds the content. 40–100 / 200–600 hours. Reuse receiver panels and PEFT; this is lower priority because translator training can obscure the quantity. **Scores: 3/3/3/4/4.**

### F09 — Test whether optimization-path information is real or arbitrary

**Abstract claim.** Only newly acquired functional distinctions along optimization predict final storage; geometric distance traveled can grow without any increase in necessary correction information.

**Closest repo work / difference.** The initial gradient, Fisher and correction-spectrum searches observe local geometry. Behavioral trajectories now save checkpoints but have not isolated path dependence at a matched endpoint.

**Definitions.** Functional path length is a sum of distances between checkpoint predictive distributions on a fixed query distribution. Fisher length is the corresponding infinitesimal metric when its assumptions hold. Neither is an information code by itself. A functional innovation is a distinction that remains useful at the endpoint, measured by intervention or conditional predictive coding, not by summing positive loss changes.

**Experiment.** Add reversible learning/unlearning cycles, change learning-rate schedules and reorder modules while matching the final learned function and utility. Compare geometric length, cumulative predictive-code regret and final decoded-file rate. Use exact small-model functions first, then Qwen/Mistral schema tasks and a held-out family. Include loops wholly within a function-equivalence class and paths that forget a previously learned task.

**Signature / falsifier.** Loops inflate path length without inflating endpoint rate; only nonredundant innovations survive a prospective prediction test. If even the innovation construction changes under harmless curriculum order, reject it as a universal variable.

**Useful failure.** A clean invariance counterexample rules out a whole family of sophisticated-looking path integrals. A positive length correlation without these controls is not worth pursuing.

**Novelty.** [The Complexity Dynamics of Grokking](https://arxiv.org/abs/2412.09810) already relates optimization stages and compression. [EDL](https://arxiv.org/abs/2601.04728) already depends on the learning algorithm. The added contribution would be a path-invariance test and a formal separation, not another training-curve plot.

**Theory / resources.** Show that any positive arc-length functional can be made arbitrarily large by repeated closed loops while endpoint description length stays fixed; identify assumptions under which a regret quantity avoids that problem. 20–60 / 100–300 hours. Reuse `on_update` and trajectory scoring. **Scores: 3/3/4/5/3.**

### F10 — Information storage and rule discovery separate

**Abstract claim.** Equal-complexity target functions can require very different learning effort while converged functional storage remains comparable, producing distinct discovery and storage regimes.

**Closest repo work / difference.** The sum/product and known-sharing experiments confounded rate with training failure. Here that failure becomes a deliberately controlled axis, with separate estimands for acquisition probability and post-acquisition rate.

**Definitions.** Target complexity is fixed under a public source distribution or DSL; acquisition means passing independently generated rule tests, not memorizing rows. Learning effort is updates/tokens to that gate. Storage rate is measured only for acquired targets, while acquisition failures remain visible in unconditional success and resource curves.

**Experiment.** Use the same underlying tables/rules with direct versus compositional presentations, hidden versus revealed grouping keys, and curricula that expose or conceal sharing. Hold source entropy, answer distribution and train/test domains fixed. Scale module count and interaction order separately. Tiny transformers establish known limits; Qwen/Mistral and a sealed family test the same intervention with equal compute.

**Signature / falsifier.** Presentation changes acquisition time greatly but converged rates little; rates drop when a rule is discovered rather than when likelihood merely improves. If learned functions stay functionally different or rates change with presentation despite matched output behavior, the storage/discovery separation is incomplete.

**Useful failure.** A phase diagram identifies where source information ceases to predict a learnable neural correction. This is stronger than filtering failed seeds out of a scaling fit.

**Novelty.** [Grokking complexity](https://arxiv.org/abs/2412.09810) already distinguishes memorization and simpler solutions; [A Compression Perspective on Simplicity Bias (2026)](https://arxiv.org/abs/2603.25839) predicts feature-selection transitions. The contribution must isolate receiver-relative correction rate at matched function, not rediscover delayed generalization.

**Theory / resources.** Analyze a two-hypothesis learner where rule discovery is slow but the rule's optimal code remains short; derive acquisition-conditioned versus unconditional rate curves. 30–80 / 150–400 hours. Reuse known-source/program datasets and their fit gates. **Scores: 3/4/4/5/4.**

### F11 — Program identity versus parameter payload

**Abstract claim.** When a receiver already implements a public family of algorithms, selecting one costs a program-identity message, whereas changing its internal table costs payload information with a different scaling law.

**Closest repo work / difference.** The four-candidate algorithm-selection campaign found almost no reliable switching and no targeting benefit. This does not retry its receiver-targeting scalar. It first verifies that a receiver can execute every candidate and then randomizes selection and payload independently.

**Definitions.** Draw program identity (J\) from a finite, explicitly coded library and payload (Z\) independently; record (H(J)\), (H(Z\mid J)\), and functional distinguishability under test queries. Library size alone is not information if many programs agree on the query distribution. Charge the library if it is not public receiver side information.

**Experiment.** Libraries of 8/64/512/4096 verified arithmetic or schema programs; payload 0/8/64/512 bits. Compare instruction-based selection, diagnostic examples and parameter updates, with equal input exposure. Qualify execution before testing selection. Develop on Qwen/Mistral; transfer the frozen decoder/protocol to Olmo and new DSL constructors.

**Signature / falsifier.** Selection rate grows near the distinguishable program-index cost while payload rate grows with missing table entries. If the receiver cannot switch even with an explicit correct index, it has not supplied the assumed interpreter, and the selection result fails.

**Useful failure.** Show that an apparent “latent algorithm library” inferred from candidate likelihoods is not actually available for controlled reuse. This directly strengthens the existing negative.

**Novelty.** [Simplicity Bias (2026)](https://arxiv.org/abs/2603.25839) already models competing features with two-part descriptions; [knowledge-capacity work](https://arxiv.org/abs/2404.05405) already varies factual payload. The new test crosses independently randomized identity and payload under an audited receiver library.

**Theory / resources.** Bound message length by the query-distinguishable program entropy plus conditional payload rate; use Fano-type bounds with the exact finite task family. 25–70 / 120–350 hours. Reuse program-selection scoring and exhaustive candidate evaluation. **Scores: 3/3/4/5/4.**

### F12 — Shared corrections create subadditive storage

**Abstract claim.** The storage cost of learning several behaviors is governed by their shared corrective structure, so joint correction codes can be much smaller than the sum of separately trained codes.

**Closest repo work / difference.** Existing within-corpus diversity and mixed task panels do not intervene on shared latent modules while measuring joint versus separate functional rate.

**Definitions.** Let (B(A)\), (B(B)\) and (B(A,B)\) be rates under the same fixed decoder and per-task distortion constraints. The empirical saving (B(A)+B(B)-B(A,B)\) is not automatically mutual information: optimization and joint interference contribute. For a declared probabilistic source, conditional information supplies a theoretical comparator.

**Experiment.** Generate paired behaviors sharing 0–100% of transformation modules while matching marginal difficulty and sample counts. Train jointly, sequentially and separately; include separate adapters with a charged task selector and merged/distilled adapters. Test overlap 4–4096 components, rank and precision. Qwen/Mistral development, sealed code/schema tasks and new-family transfer.

**Signature / falsifier.** Joint rate saving increases with true shared modules and generalizes to held-out compositions. If sharing merely improves training fit or joint training sacrifices one task, the information-sharing claim fails.

**Useful failure.** Superadditive costs under verified shared functions reveal interference or access overhead; that is a useful counterexample to treating adapter bits as a modular information inventory.

**Novelty.** [Data Mixing Optimization for SFT (2025)](https://proceedings.mlr.press/v267/li25bh.html) models task transfer through training data, while [SkillPacks](https://arxiv.org/abs/2505.18502) builds reusable skills. This measures a jointly constrained communication cost with randomized common structure.

**Theory / resources.** For a common source component (U\) and independent private components, derive an additive common/private code and conditions for subadditivity. Show why a restricted neural decoder can violate it. 30–70 / 160–450 hours. Reuse datasets, merger/codec primitives and the common trainer. **Scores: 4/4/4/4/4.**

### F13 — Average distortion hides expensive rare corrections

**Abstract claim.** A 90%-gain budget can discard an entire rare but important learned behavior; stratified functional rate exposes a different and predictable storage requirement.

**Closest repo work / difference.** The new (B^*\) improves the codec but still uses an average likelihood threshold. High-gain and rewrite controls do not isolate rare correction prevalence at fixed complexity.

**Definitions.** Compare average distortion, maximum group distortion and a tail-risk criterion such as CVaR. Groups are fixed by the generator or application requirements before evaluation. “Important” is an explicit task requirement, not a weight chosen after seeing a failure. Report each group's actual utility and rate.

**Experiment.** Embed independent correction modules at query probabilities from (10^{-1}\) to (10^{-4}\); oversample evaluation with importance weights so rare groups have precise estimates. Keep total unique source information fixed while varying mass. Reuse synthetic programs, then code corner cases and schema exceptions on Qwen/Mistral plus a held-out family. Compare training balancing with identical evaluation distributions.

**Signature / falsifier.** Average-budget curves become flat as rare modules drop below allowed distortion, while per-group budgets retain their cost. If learned rare behavior survives all low-rate codes, the proposed rate-tail separation does not occur for that family.

**Useful failure.** Even a null bounds the vulnerability of the existing 90% target. A positive result can justify replacing one scalar with a requirement-indexed rate surface.

**Novelty.** [EDL](https://arxiv.org/abs/2601.04728) already explains why rare structure contributes little to expected gain. [Prompt rate–distortion](https://arxiv.org/abs/2407.15504) already depends on the query distribution. Novelty requires decoded-file evidence that average retention discards specified learned capabilities and a successful tail-aware forecast.

**Theory / resources.** A two-group source gives an explicit threshold: if allowed distortion exceeds a group's weighted error cost, the optimal average code may spend zero bits on it. 10–30 / 80–200 hours; reuse all scorers and add stratified reduction rather than new model machinery. **Scores: 3/5/5/5/4.**

### F14 — A representability transition, not a rank sweep

**Abstract claim.** Adapter architecture changes correction rate chiefly when it crosses a functional representability threshold, with excess rank beyond that threshold contributing implementation overhead rather than new behavior.

**Closest repo work / difference.** Budget-matched rank and the wider sketch already show rank matters. This specifies a known task rank/interaction structure and predicts a discontinuity before tuning, instead of fitting rank effects across natural tasks.

**Definitions.** In a controlled frozen-feature model, let the target residual map have algebraic rank (r_f\). This is a property of the function on a fixed domain, not the trained LoRA factors. Adapter rank, location and nonlinearity define which maps are reachable. For full LLMs, do not equate a data matrix rank with a global representability theorem.

**Experiment.** Construct rank-1 through rank-64 residual maps with matched norm, entropy and labels; compare linear-head, selected-layer and distributed LoRA corrections. Sweep around predicted thresholds, with converged training and independent distortion tests. Establish exact results on small frozen-feature models, then test the qualitative threshold on Qwen/Mistral and new-family tasks with controlled interaction order.

**Signature / falsifier.** Acquisition or achievable rate changes at the predicted boundary, while increasing container rank beyond it leaves functional rate stable after recoding. A gradual optimization-dependent effect without the predicted threshold rejects the representability explanation.

**Useful failure.** It identifies where a linear correction model ceases to describe nonlinear LLM adaptation.

**Novelty.** [GeLoRA (2024)](https://arxiv.org/abs/2412.09250) estimates rank from representation dimension; [LoRA-Squeeze](https://arxiv.org/abs/2602.10993) anneals/compresses rank. The proposed distinction is a ground-truth functional threshold with matched source information. Without that construction, this is incremental.

**Theory / resources.** Rank constraints yield an exact impossibility/approximation bound in the frozen linear setting. 25–70 / 120–350 hours; reuse PEFT attachment choices, balanced SVD and known-program data. **Scores: 3/4/4/5/3.**

### F15 — Singular learning theory must predict code lengths, not correlations

**Abstract claim.** In a controlled adaptation family, singular model structure predicts a finite-sample coding term that transfers across parameterizations and forecasts a measured rate–distortion curve.

**Closest repo work / difference.** The broad Fisher/log-volume search is already exhausted. This is only distinct if a solvable toy model supplies the coefficient and functional form before fitting the neural panel.

**Definitions.** Separate stochastic complexity/learning coefficients from actual lossy adapter bytes. An asymptotic evidence term such as (\lambda\log n\) describes a probabilistic coding problem with a specified prior; it is not automatically the rate needed to retain 90% of a trained model's gain.

**Experiment.** Begin with a singular factorized linear model where the target function and prior are explicit. Derive the predicted term; test exact or high-quality numerical coding. Only then evaluate functionally matched small transformers and Qwen/Mistral adapters, with a held-out architecture. Cross sample count over at least two decades and compare constant-overhead, logarithmic and power alternatives on prospective sizes.

**Signature / falsifier.** A parameterization-invariant coefficient predicts new curves within uncertainty. If the coefficient depends on arbitrary prior/temperature choices or only correlates after calibration, stop.

**Useful failure.** A precise demonstration that stochastic complexity and post-training functional storage are different objects is useful; another selected LLC-versus-size scatterplot is not.

**Novelty.** [Compressibility Measures Complexity: MDL Meets SLT (2025)](https://arxiv.org/abs/2510.12077) already connects learning coefficients with compression. The only defensible extension is a derived prospective code-length prediction in a different operational problem.

**Theory / resources.** Establish the mapping—or prove a separation—between Bayesian stochastic complexity and distortion-constrained message length in the toy model. 40–120 / 250–700 hours, with substantial theory work. Reuse adapters and codecs; do not build an LLC estimator without a theorem target. **Scores: 2/2/2/4/3.**

### F16 — High risk: the missing variable is execution cost

**Abstract claim.** The apparent information cost of adaptation is often the cost of making a correction cheap to execute, so no receiver-independent storage law exists without a compute constraint.

**Closest repo work / difference.** Container/rank effects and failed known-program scaling suggest this, but the project has not measured a rate–execution-time surface for the *same function*. This deliberately abandons the search for one information scalar explaining LoRA bytes.

**Definitions.** (B(d,\tau_{\rm setup},\tau_{\rm query})\) minimizes message bits under distortion, decoder setup time and per-query work limits. A short program may need expensive computation; a compiled table may need many bits. Those are two implementations of one function, not different source entropies.

**Experiment.** Use public interpreters, known arithmetic programs, lookup tables and neural adapters implementing identical correction families. Scale program depth, input domain and repeated-query count independently. Compare external execution, replay-trained adapters and direct compiled corrections with all code and compute charged. Natural validation uses code transformations with known tests; Qwen/Mistral development and new-family confirmation. Do not let an external solver secretly define the whole neural task as solved.

**Signature / falsifier.** At fixed function and utility, tightening query-time limits raises the best achieved byte cost, and different receivers trace different surfaces. If a short constant-cost decoder attains all observed functions, implementation-time pressure does not explain their large adapters.

**Useful failure.** A demonstrated small executable correction undercuts the claim that adapter bytes measure necessary information; either outcome changes the project's object.

**Novelty.** [Conditional task complexity](https://arxiv.org/abs/2602.15829) already counts adaptation descriptions, and [usable information](https://arxiv.org/abs/2002.10689) already restricts the observer. The prospective contribution is a measured compute-conditioned rate surface and an explicit separation theorem. Merely invoking time-bounded Kolmogorov complexity would be empty.

**Theory / resources.** Prove a time/space tradeoff for a restricted query model or fixed interpreter; do not claim an unconditional lower bound on arbitrary neural computation. 60–150 / 400–1,200 hours. Reuse program-selection tasks and adapters; this needs more new infrastructure than the leading options. **Scores: 5/3/2/4/5.**

## Full design 1 — Can learned predictive information price a correction?

### Paper thesis and scope

The strongest version is: **an explicitly measured learning-side information curve predicts a function-side message budget under a fixed resource contract, with known and tested exceptions caused by receiver implementation and incomplete learning**. Do not claim the quantity is intrinsic to a dataset, a parameter tensor, or all possible learners.

The natural-task estimator must be computable without a generating program. Start with the existing online/prequential learner and record its full prefix loss curve, including fresh-block prediction and independent population-loss estimates. Use the *same* instrument on controlled and natural tasks. The synthetic generator supplies an external calibration standard; it must not supply hidden features to the natural-task predictor. Lock a small set of theory-derived curve families and a model-averaged uncertainty rule on development data. If the early prefix cannot identify the curve's eventual information, output a wide interval or abstain; do not silently use the final fine-tune loss.

This is not a claim that EDL equals adapter bytes. The central experiment tests that proposed bridge. If it requires a different arbitrary scale factor for every receiver, or only works after selecting a favorable code family, the desired paper has failed.

### Minimum decisive experiment

1. **Calibrate the instrument without an LLM.** Use a public family of independent binary modules, a partially informed Bayesian receiver and an exact interpreter. Verify actual prefix-free encoding/decoding and empirical distortion against the analytic curve. Scale unknown components over 4–4,096. Count indexes when the query sequence is not public. This catches measurement errors cheaply.
2. **Test neural implementation.** Use two small transformer architectures at six logarithmically spaced complexities and two receiver-overlap levels. Require high rule-test competence at the declared endpoint. Compare actual balanced-rank adapter files, a replay message and the public function-code baseline. Include at least three task draws at each setting, rather than treating seeds on one function as independent functions.
3. **Do one genuinely prospective neural forecast.** Develop on Qwen/Mistral with two short-output task families; freeze the predictor and code menu, then forecast one new task family before its full training. Use 8/32/128-update observations only; the forecaster chooses its observation cutoff on development data. This is a first falsifier, not yet the new-family headline.

Preregister the pilot stop: if functional equivalence cannot be checked, fewer than two decades of (K\) can be learned, actual-file uncertainty spans more than a factor of two for most cases, or a global mapping fails even the first untouched family, do not scale F01. Return the result to F02/F16 rather than searching 50 more summaries.

### Figure-by-figure story

| Figure | Experiment and purpose |
|---|---|
| 1 | Same target, different bytes: old rank-mask, informed rank code, functional code and replay; actual files with uncertainty. Explain which operational problem each rate solves. |
| 2 | Known missing information over three orders of magnitude, randomized receiver overlap. Plot analytic rate–distortion envelope and actual decoded neural rates, with learning failures visible. |
| 3 | Learning versus storage: prefix information curves and final task-rate curves; distinguish matched entropy with different learnability, and matched behavior with different implementation overhead. |
| 4 | Prospective budget forecasts: predicted versus purchased/tested bytes, identity line, log-byte errors, coverage and abstentions. Separate unseen task, unseen receiver and unseen family panels. |
| 5 | Parameterization and resource controls: rank/attachment, replay versus adapter, decoder compute and functionally equivalent recodings. Show where the law remains valid and where it breaks. |
| 6 | Natural-task confirmation with executable outputs; utility achieved at the forecast budget, alongside corpus-mean and fixed-budget baselines. No test-specific intercepts. |

### Broader confirmation matrix

Keep the initial matrix small enough to complete. Development: Qwen2.5-7B and Mistral-7B, two verified short-output task families, three independent function draws, two exposure levels, endpoint rank 16. Confirmation: two wholly new task structures and an unused family, with fixed early measurement and forecasting code. Add ranks 4 and 32 and one different adapter placement **only after** the baseline forecast passes. These are tests of parameterization dependence, not extra points for a pooled regression.

The headline prediction should be evaluated over at least 24 independently defined task/receiver cells and several function draws; 24 near-identical adapters are not 24 tasks. Use cluster bootstrap by task draw and receiver as appropriate. Primary suggested gates, to be frozen after a development-only power check: log2-byte MAE at most .5 bits; at least 20% lower error than the locked corpus/nearest-task baseline; absolute median log2 bias at most .15; nominal 90% prediction-interval coverage with median width no greater than a factor of two; and at least 90% of purchased-code evaluations satisfying their stated utility target, subject to the registered uncertainty procedure. Show every component rather than combining them into one “pass.”

Confidence in a code passing must use independent testing. Calibration can choose the code using a one-sided utility bound; evaluation then reports success on a sealed sample. If no code passes, report an unattained requirement. If the smallest code passes, report a left-censored optimum. A coarse menu may only identify a rate bracket, which is still a useful deliverable.

### Formal result worth proving

Begin with a solvable benchmark, then attempt the bridge. For independent unknown fair bits queried with probabilities (p_j\), a Bayesian learner's expected online label code after (n\) samples is

\[
J_n=\sum_j[1-(1-p_j)^n],
\]

and its expected residual population log loss is

\[
u_n=\sum_jp_j(1-p_j)^n.
\]

The corresponding expected excess code is (J_n-nu_n\), approaching the number of unknown bits with enough data. These occupancy identities are a calibration model, not a new universal neural theorem. At full acquisition, independent-component rate–distortion gives (K[1-h_2(d)]\). The research theorem should state **additional conditions** under which a receiver/decoder realizes that rate within an explicit overhead, or construct equal-learning-code examples with arbitrarily different restricted implementation costs.

For natural tasks, the generator is unknown and the early curve may not identify (J_\infty\). The formal result should make that uncertainty explicit. No theorem about a Bayesian lookup learner establishes a power law for LoRA-trained LLMs.

### Cost, reuse and decision

Budget 60–120 H100-hours for the neural pilot and 300–900 for full confirmation; exact synthetic coding is CPU work. At a 500-hour cap, shorten output tasks and test a sparse, calibration-selected code menu before broadening architecture. The live trajectory campaign shows why: full GSM generation across many codes takes roughly 7–10 H100-hours per trajectory cell, while its training often takes under one hour. Count evaluation first.

Reuse the current codec and training callback. The only required measurement changes are the actual-file reducer, explicit data roles, and the predictive-code logging already implied by the prequential experiment. Do not create another framework or classifier zoo.

## Full design 2 — The observation budget for forecasting a fine-tune

### Paper thesis

**The predictability of a learned correction is a function of what the forecaster has observed about learning.** A failed zero/one-step prediction can reflect endpoint uncertainty, an incorrect forecast model, or a bad compression target. This paper separates those explanations and, if successful, identifies a cheap observation level that supports calibrated forecasts on new tasks/families.

Do not assert that prediction before tuning is logically impossible. Given the full deterministic dataset, optimizer, seed and unlimited compute, the endpoint can be calculated at “time zero” by running the optimization. An impossibility statement therefore needs an explicit observation or computation restriction. Random future data order supplies real conditional uncertainty only if it has not already been supplied to the forecaster.

### Minimum decisive experiment

Use 12 development task/receiver cells spanning high/low gain and simple/shared corrections. Save (k=0,1,8,32,128\) during one common schedule. At two prefixes, fork four independent continuations with newly drawn future order/seed; retain unchanged-prefix, deterministic-repeat and known-late-reveal controls. Predict the final *distribution* of achievable functional budgets before decoding endpoint files. This separates variability among continuations from uncertainty due to a coarse rate grid.

Three forecasters are enough: (a) corpus/nearest-task baseline; (b) the current early-budget curve, with a single development-fitted calibration map; (c) a small functional-prediction model of which correction will be acquired. An endpoint-informed oracle measures how much forecast error could in principle be removed, but never appears as a prospective predictor. Lock all feature choices before the fresh test cells. Evaluate calibration, not just ordering.

Stop if endpoint functional behavior is unstable even at the full training gate, if the code budget is unresolvable at the needed precision, or if the cheapest baseline matches the prefix methods. Do not broaden a failed scalar under a new name.

### Figure-by-figure story

| Figure | Experiment and purpose |
|---|---|
| 1 | Reproduce the current one/eight-step ranking/calibration separation using actual file budgets. Show why better Spearman is not the desired result. |
| 2 | Forked training continuations: same prefix, different endpoints; display conditional spread in learned function and rate, not just weights. |
| 3 | Forecast error and interval width versus observations, data inspected and H100-time. Include an explicit cost of the probe. |
| 4 | Early-indistinguishable controlled tasks with later branch selection; test the predicted observation lower bound. |
| 5 | Sealed forecasts on new tasks, receivers, a new family and rank/placement changes; identity calibration and purchased-code utility. |
| 6 | Failure taxonomy: unresolved learning, codec uncertainty, forecast misspecification and stable predictable cases. A precise boundary is the result. |

### Confirmation matrix and locked decisions

Develop on the old Qwen/Mistral tasks and new instances; confirm on at least two unused program/data structures plus an unused family. Randomize task draws independently of optimizer seeds. Cross predictable versus delayed-rule families with two future training budgets and ranks 4/16/32, but use a fractional matrix so the same checkpoint series supports several comparisons. Cap prefix observation at a registered fraction of full training compute, suggested 10%; a forecaster costing almost as much as fitting the final adapter is a different product.

Use F01's calibrated-prediction metrics and intervals. Add the smallest observation budget reaching a predeclared risk target, with uncertainty over that threshold. Do not choose (k\) on each test receiver. Compare a single globally frozen (k\) with a preregistered adaptive stopping rule based only on prefix uncertainty. Track abstentions and their cost.

### Formal result worth proving

For (Y=\log_2 B\) and observation (O_k\), squared-error risk has the exact decomposition

\[
\mathbb E[(\hat Y-Y)^2]
=\mathbb E[\operatorname{Var}(Y\mid O_k)]
+\mathbb E[(\hat Y-\mathbb E[Y\mid O_k])^2].
\]

Forks estimate a component of the first term under the declared random continuation process; they do not estimate all uncertainty over tasks or all possible forecasters.

For two worlds with endpoint log-budget separation (\Delta\) and observation laws (P_0^k,P_1^k\), a standard two-point argument gives a minimax squared-error lower bound of at least

\[
\frac{\Delta^2}{8}\left(1-\operatorname{TV}(P_0^k,P_1^k)\right).
\]

If the worlds are identical until a diagnostic event occurring with probability (q\) per independent observation, the remaining indistinguishability is at least ((1-q)^k\). Thus detecting it with miss probability at most (\delta\) needs approximately (\log(1/\delta)/q\) observations. The theorem applies to the defined query/prefix access model. It does not prove a universal minimum number of SGD steps from full dataset access.

The worthwhile extension is to show that a concrete prefix observable approaches this bound in a nontrivial learned correction family, then test its practical forecast boundary on LLMs.

### Cost, reuse and fallback

40–100 pilot / 200–600 confirmation hours. Fork only at selected prefixes; do not multiply every task by every (k\), since a single trajectory supplies all prefix observations. Reuse `budget_probe`, `on_update` and current trajectory checkpoints. If no cheap observation level works, the paper must present a controlled prediction limit with constructive examples and useful uncertainty estimates; a broad failed-regression survey is not enough.

## Full design 3 — Compression of improvement versus compression of damage

### Paper thesis

**The full fine-tune is an unreliable reference for “necessary learned information.”** A training run can improve teacher-forced likelihood while adding harmful behavior. Useful-task preservation, imitation of the checkpoint and removal of damage therefore have different rate frontiers. The existing Qwen recovery is strong motivation, but the .3 shrinkage control prevents claiming that quantization discovers a unique semantic filter.

### Minimum decisive experiment

Finish the current trajectory/recovery evidence before another campaign. On the saved three Qwen permuted adapters, freeze one code and one scalar-shrinkage rule using calibration only. Evaluate on new, verifier-backed problem templates absent from GSM8K/GSM-Symbolic selection. Separate baseline-correct and baseline-incorrect strata using an independent qualification sample and preserve unconditional totals.

Run the same fixed comparisons on Mistral and one new family with qualified base headroom. Minimum candidate set: base, raw full adapter, rank-1 binary, rank-1 BF16, scale-.3 rank-1 BF16, norm-matched rank-1, full-update shrinkage and matched random truncation. The .3 value is an existing development choice, not a newly discovered prospective optimum. Include a calibrated scalar-search baseline so a lucky fixed value does not understate shrinkage.

The decisive question is whether the compact code both retains newly acquired correct behavior and repairs damage, beyond what a simple scale adjustment explains. A return to base accuracy without retained acquisitions does not show information-rich compression.

### Figure-by-figure story

| Figure | Experiment and purpose |
|---|---|
| 1 | Paired per-problem transitions: base→raw→compressed. Distinguish acquired successes, preserved successes, damage and repair, with original denominators. |
| 2 | Two-dimensional acquisition/damage frontiers versus actual bytes. Overlay scalar shrinkage and random truncation rather than hiding them in an appendix. |
| 3 | Training-time evolution: when useful behavior and damage emerge, using the already locked trajectory checkpoints. Show incomplete or unattained targets. |
| 4 | Controlled corruption and receiver headroom: same content, randomized supervision corruption, matched schedules. Separate receiver identity from target-program identity. |
| 5 | Sealed task/family transfer of the selected intervention. Full generated outputs and execution checks; no per-test-code selection. |
| 6 | Rate target reversal: compare likelihood-gain, checkpoint-imitation and externally fixed utility requirements. Show the practical cost of buying the wrong budget. |

### Confirmation matrix

Use three supervision levels—aligned, partially corrupted, permuted—on one controlled task family, then two natural task families with executable scoring. Cross receiver competence rather than merely adding a larger model: one low-headroom acquisition receiver, one strong receiver vulnerable to damage, and an untouched family qualified without viewing treatment effects. Three seeds per chosen cell; keep exact source problems shared across arms and distinct across calibration/test.

Controls must include fixed-task precision/recall or group utilities if exact match can hide damage. For SQL, use execution equivalence; normalized string equality is insufficient. For code, hold the test suite fixed and include unseen inputs. If acquisition/damage partitions vary under stochastic base generation, estimate them from independent repeated base samples rather than conditioning on a lucky single answer.

Suggested go/no-go: at least a 5-point useful-utility improvement over the raw adapter on a sealed task, with a positive paired interval; retain at least 90% of independently measured acquisitions; report the difference from the best calibration-selected shrinkage baseline. If that last difference is indistinguishable from zero, publish the target-mismatch mechanism only if it is broad and well predicted—do not call binary coding uniquely effective.

### Formal result worth proving

Construct functions (f_{u,v}\) with independent useful component (u\) and harmful or task-irrelevant component (v\). An imitation distortion penalizes losing either component; task utility rewards (u\) and may penalize (v\). By varying the entropy/implementation of (v\) at fixed (u\), imitation rate can grow while useful-task rate remains fixed. Conversely, a small change in average loss can encode a hard-to-implement rare correction. This yields a clean separation, not an assertion that neural singular vectors align with (u,v\).

The empirical mechanism must then test whether a specific code changes acquisition and damage differently, while norm/shrinkage controls distinguish direction selection from amplitude reduction. A nonmonotonic curve for one quantization path is possible even though the *best achievable* utility envelope cannot worsen as the allowed code menu expands with budget.

### Cost and stopping rule

10–35 hours for decisive reuse and fresh scoring; 80–240 for a focused confirmation, potentially more if long generations dominate. Reuse the ongoing H100 evaluations before duplicating them. If scaling alone explains recovery and the target mismatch is limited to one damaged Qwen setting, treat this as an important corrective result within F01 rather than stretching it into the main paper.

## What I would not spend the next allocation on

Another Fisher/effective-rank scalar search; a wider early LoRA sketch; an unqualified sum/product comparison; a family-dependent fitted intercept; a fit across five nearby dataset sizes labelled a scaling law; a nominal-bitwidth plot; or a generic high-rank-then-compress result. Each either repeats a retained failure or overlaps directly with existing work.

The inexpensive first commitment is measurement, followed by a small duality test and a sealed forecast. The strongest outcome may be a scoped law with an explicit implementation gap, not the original universal (I,N,P\) power law.
