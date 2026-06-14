# Leveraging TabPFN for counterfactual explanation generation via structural causal models

**A Prior-Data Fitted Network can be repurposed from classification to counterfactual generation by training on SCM-simulated factual–counterfactual pairs, enabling amortized, causally consistent counterfactual explanations in a single forward pass.** This represents an unexplored intersection: while CausalPFN and Do-PFN have extended PFNs to causal effect estimation, and generative methods like C-CHVAE and CeFlow produce counterfactual explanations via VAEs and flows, no published work combines the PFN in-context learning paradigm with counterfactual explanation generation. The architecture proposed here—CounterPFN—bridges this gap by pre-training a dual-attention transformer on millions of synthetic SCM-derived counterfactual triplets (factual instance, intervention, counterfactual outcome), learning to produce causally grounded counterfactual explanations at inference time without per-instance optimization.

---

## TabPFN encodes an implicit Bayesian ensemble in a single forward pass

TabPFN is a transformer that performs classification (and regression, in v2+) by treating the entire training set as an in-context "prompt." Rather than fitting parameters to a specific dataset, TabPFN was pre-trained on **~130 million synthetic datasets** (v2) sampled from a prior over data-generating processes, learning a general-purpose learning algorithm encoded in its weights.

**Architecture.** TabPFN v2 uses a **dual-attention transformer** with 12 layers and ~11M parameters. Each cell in the input table receives its own embedding, forming an N×P×E tensor (rows × features × embedding dimension of 192). Each transformer layer applies two attention operations sequentially: *feature attention* (row-wise), where cells attend to other cells in the same data point to capture within-sample feature relationships, and *datapoint attention* (column-wise), where cells attend to the same feature across different samples to learn cross-sample patterns. An asymmetric attention mask ensures test tokens attend only to training tokens, never to other test tokens—preventing information leakage. The latest version, **TabPFN v2.5**, scales to 24 layers, 50,000 samples, and 2,000 features, adding 64 learned "thinking rows" that provide extra computational capacity.

**In-context learning.** At inference, the entire labeled training set and unlabeled test points are fed together. The transformer's attention mechanism routes information from labeled examples to test queries, producing calibrated class probabilities in a single forward pass—**no gradient updates, no hyperparameter tuning**. This is functionally equivalent to a meta-learning system where the "inner loop" adaptation is performed entirely through attention.

**Bayesian interpretation.** A PFN trained to the global optimum provably approximates the **posterior predictive distribution** (PPD): p(y|x, D) ∝ ∫ p(y|x,ϕ) · p(D|ϕ) · p(ϕ) dϕ. In practice, a single forward pass implicitly marginalizes over all hypotheses in the prior—every possible SCM, every BNN architecture—weighted by their posterior probability given the observed data. This gives TabPFN well-calibrated uncertainty estimates and performance that rivals tuned AutoML ensembles on datasets up to ~10,000 samples, with **over 5,000× speedup**.

**The prior is the key innovation.** TabPFN's prior mixes two generative families. The *SCM prior* samples random directed acyclic graphs structured as layered MLPs with stochastic edge dropout (Beta(0.1, 5.0)-distributed), applies random nonlinear activation functions (ReLU, Tanh, ELU, Identity, or Threshold), and propagates additive Gaussian noise through the graph. Feature nodes and a target node are randomly selected from the DAG, meaning features can be causes, effects, or colliders relative to the target. The *BNN prior* samples random neural network architectures with random weights. Continuous targets are converted to classification labels via random binning. Critically, nearly all hyperparameters—number of layers, hidden nodes, noise magnitude, edge density—are themselves drawn from meta-distributions, so the PPD integrates over this entire hyperparameter space.

---

## SCMs provide the formal machinery for counterfactual reasoning

A **Structural Causal Model** M = ⟨U, V, F, P(U)⟩ consists of exogenous variables U (unobserved background factors), endogenous variables V (observed), structural equations F where each Vᵢ := fᵢ(PAᵢ, Uᵢ), and a distribution P(U) over the exogenous noise. The structural equations encode *mechanisms*—autonomous, modular causal relationships that can be independently modified.

Pearl's **Ladder of Causation** stratifies causal reasoning into three strictly hierarchical levels. Rung 1 (association) handles observational queries P(Y|X). Rung 2 (intervention) handles do-calculus queries P(Y|do(X=x)), which sever incoming edges to X. Rung 3 (counterfactual) handles individual-level what-if queries: "Given that this person was denied a loan, would they have been approved had their income been $10K higher?" This requires the three-step **abduction-action-prediction** procedure:

1. **Abduction**: Infer the exogenous noise u = F⁻¹(x^F) from the observed factual instance x^F, recovering the individual's specific background conditions.
2. **Action**: Modify the SCM by replacing structural equations for intervened variables with constants, creating a modified model M_a.
3. **Prediction**: Propagate the original noise u through the modified model to obtain the counterfactual x^CF = F^a(u).

This procedure is uniquely powerful because it preserves the individual's identity (same noise terms) while exploring alternative scenarios—precisely what counterfactual explanations require. The connection to TabPFN's prior is direct: **TabPFN already generates training data from SCMs**, sampling DAGs, structural equations, and noise. Extending this to generate *paired* factual and counterfactual instances is a natural step.

---

## Existing counterfactual methods reveal a clear gap for PFN-based generation

The counterfactual explanation literature has converged on key desiderata: **validity** (the prediction must change), **proximity** (minimal change from the original), **sparsity** (few features changed), **plausibility** (the counterfactual should be realistic), and **causal consistency** (changes must respect causal dependencies). No existing method optimally satisfies all simultaneously.

**Optimization-based methods** (Wachter et al., DiCE, FACE) solve a per-instance optimization problem balancing validity against proximity. DiCE adds diversity via determinantal point processes. FACE ensures counterfactuals lie on high-density paths through the data manifold. These methods are flexible but slow—each new instance requires fresh optimization—and typically ignore causal structure.

**Generative methods** amortize the cost. C-CHVAE embeds counterfactual search in a VAE's latent space, naturally ensuring plausibility. CeFlow uses normalizing flows with class-conditional Gaussian mixture priors for deterministic, invertible generation. HyConEx (2025) simultaneously classifies and generates counterfactuals in a single forward pass using invertible flows. The emerging TDCE and SCD use diffusion models with classifier guidance for tabular counterfactuals. These methods are fast at inference but require separate training per dataset and rarely enforce causal constraints natively.

**Causally-aware methods** (Karimi et al., CARMA) use SCMs to generate counterfactuals via the abduction-action-prediction process. CARMA uses deep causal generative models (normalizing flows) to approximate unknown SCMs, achieving amortized causal counterfactual inference. However, these methods assume a fixed, known (or estimated) causal graph for each specific dataset.

**The gap**: No existing method combines (a) PFN-style pre-training on diverse synthetic SCMs, (b) in-context learning that adapts to new datasets without retraining, and (c) causally grounded counterfactual generation. TabPFGen has demonstrated TabPFN can be repurposed as an energy-based generative model via SGLD sampling, and FairPFN uses PFNs for counterfactual fairness—but neither produces counterfactual explanations. Miller and Schölkopf (2025) showed transformers can perform in-context counterfactual reasoning by learning "noise abduction heads." CausalPFN and Do-PFN extend PFNs to causal effect estimation but do not generate individual counterfactual instances.

---

## CounterPFN: a proposed architecture for in-context counterfactual generation

The core idea is to train a PFN-style transformer that takes as input a labeled dataset, a query instance x^F with its factual prediction, and a specification of which features to intervene on—and outputs a counterfactual instance x^CF that (a) achieves the desired prediction, (b) is minimally different from x^F, (c) respects causal dependencies, and (d) lies on the data manifold. The model learns this mapping by pre-training on millions of synthetic SCM-derived counterfactual triplets.

**Input representation.** The input consists of three components concatenated along the sample dimension of a dual-attention transformer:

- **Context set** (N labeled examples): The training data {(x₁,y₁), ..., (xₙ,yₙ)}, encoded exactly as in TabPFN v2 with per-cell embeddings and dual attention. This provides the model with the data distribution and implicit causal structure.
- **Factual query**: The instance x^F along with its prediction y^F and the desired target class y', encoded with a special token type to distinguish it from context examples. A binary intervention mask m ∈ {0,1}^P indicates which features are actionable (can be modified).
- **Counterfactual output tokens**: Placeholder tokens (initialized to x^F values) that the model fills in with counterfactual feature values through attention to the context set and factual query.

**Architecture details.** The transformer uses TabPFN v2's dual-attention design (alternating feature-attention and datapoint-attention layers) with modifications:

- **Three token types** with learned type embeddings: context tokens, factual-query tokens, and counterfactual-output tokens.
- **Attention mask**: Context tokens attend to all context tokens. Factual-query tokens attend to context tokens and themselves. Counterfactual-output tokens attend to everything—context tokens (to learn the data distribution and causal structure), factual tokens (to preserve proximity), and other counterfactual tokens (to maintain internal consistency across features, unlike TabPFN's test tokens which are isolated).
- **Intervention mask injection**: The binary mask m is embedded and added to the counterfactual output tokens' representations, signaling which features may change.
- **Output head**: A 2-layer MLP per feature type—linear output for numerical features, softmax over categories for categorical features—produces the counterfactual instance x^CF.
- **Auxiliary classification head**: A separate head predicts the class of the generated counterfactual, enabling end-to-end validity enforcement during training.

**Why dual attention is critical for counterfactuals.** Feature attention (row-wise) allows the model to capture within-instance dependencies—when generating a counterfactual, changing income should propagate to savings, tax bracket, and loan eligibility within the same row. Datapoint attention (column-wise) allows the model to learn the empirical distribution of each feature from the context set, ensuring plausibility. The combination enables the model to simultaneously respect causal structure (feature attention propagates interventions through learned mechanisms) and data manifold constraints (datapoint attention grounds each feature in observed ranges).

---

## Training CounterPFN on SCM-simulated counterfactual triplets

The training strategy extends TabPFN's prior-fitting paradigm from classification to counterfactual generation. Each training step constructs a complete counterfactual scenario from scratch.

**Phase 1: SCM-based data and counterfactual pair generation.** Each training iteration proceeds as follows:

1. **Sample an SCM** from the prior. Draw a random DAG G using TabPFN's layered-MLP skeleton with edge dropout (or Erdős-Rényi/scale-free for diversity). Sample structural equations fᵢ from a distribution over function classes: linear, MLP with random activations (ReLU, Tanh, ELU), polynomial, or additive noise models. Sample noise distributions P(Uᵢ) with meta-sampled parameters.

2. **Generate observational data.** Propagate noise through the SCM via ancestral sampling to produce N+1 instances. Designate N instances as the context set and 1 as the factual query x^F. Assign a target node and compute classification labels via random binning.

3. **Train a synthetic classifier.** Use the SCM's Bayes-optimal decision boundary (derived from the structural equations and noise distributions) as the ground-truth classifier f. Alternatively, train a simple classifier (logistic regression, small MLP) on the N context examples. This classifier defines what "validity" means—the counterfactual must change *this* classifier's prediction.

4. **Generate ground-truth counterfactuals via abduction-action-prediction.** For the factual query x^F:
   - **Abduct**: Recover exogenous noise u = F⁻¹(x^F) by inverting the structural equations (straightforward for additive noise models: uᵢ = xᵢ - fᵢ(PAᵢ)).
   - **Select intervention targets**: Randomly sample a subset of actionable features and choose intervention values that would produce the desired class change. This can be done by: (a) optimizing over intervention values to find the minimal intervention achieving class change, or (b) sampling intervention values from the target class's conditional distribution.
   - **Predict**: Propagate the original noise u through the modified SCM to obtain x^CF and the intervention mask m.

5. **Construct the training example**: Input = (context set, x^F, y^F, y', m); Target = x^CF.

**Phase 2: Multi-objective training loss.** The model is trained end-to-end with a composite loss:

**L = λ₁·L_recon + λ₂·L_valid + λ₃·L_prox + λ₄·L_causal + λ₅·L_sparse**

- **L_recon** (reconstruction): MSE for numerical features, cross-entropy for categorical features, between the predicted counterfactual x̂^CF and the SCM-generated ground-truth counterfactual x^CF. This is the primary supervisory signal and is unique to the PFN approach—**no existing generative counterfactual method has access to ground-truth counterfactuals during training**, because they train on real data where true counterfactuals are unobserved.
- **L_valid** (validity): Cross-entropy between the auxiliary head's predicted class for x̂^CF and the desired target class y'. Ensures the counterfactual changes the prediction.
- **L_prox** (proximity): Weighted L1 distance between x̂^CF and x^F, penalizing unnecessary changes. Weighted by (1-m) to avoid penalizing changes to explicitly intervened features.
- **L_causal** (causal consistency): Measures the discrepancy between the model's output and what the SCM predicts for downstream (non-intervened) features given the intervened values. Specifically: for non-intervened features that are descendants of intervened features, compute the SCM-predicted value from the counterfactual parents and the abducted noise—the model's output should match.
- **L_sparse** (sparsity): L0 approximation (e.g., straight-through estimator on feature change indicators) encouraging minimal feature changes.

**Phase 3: Curriculum and scaling strategy.** Training follows a curriculum of increasing complexity:

- **Stage 1** (epochs 1–50): Simple SCMs with 3–8 nodes, linear structural equations, Gaussian noise. Small context sets (N=50–200). Only L_recon + L_valid active. The model learns basic counterfactual generation.
- **Stage 2** (epochs 50–150): Medium SCMs with 5–20 nodes, nonlinear equations (MLPs, polynomials), mixed noise. Context sets up to N=1,000. All loss terms active. The model learns causal propagation and manifold constraints.
- **Stage 3** (epochs 150–300): Complex SCMs with 10–50 nodes, deep nonlinear equations, categorical features, missing values, class imbalance. Context sets up to N=5,000. Diverse intervention patterns (single-feature, multi-feature, constrained). The model learns robust, realistic counterfactual generation.
- **Optional Stage 4** (fine-tuning): Fine-tune on a curated set of real-world datasets where approximate causal graphs are known (e.g., from domain expertise or causal discovery algorithms), using optimization-based counterfactuals as pseudo-ground-truth. This bridges the synthetic-to-real gap.

**Key hyperparameters** for the SCM prior should be broadly distributed to maximize generalization. Graph density (edge probability) should follow Beta(2,5) to favor sparse graphs. The number of features should be sampled log-uniformly from 3 to 500. Noise-to-signal ratios should span 0.01 to 1.0. Structural equation complexity should range from linear to 3-layer MLPs with up to 128 hidden units.

---

## Why this approach has fundamental advantages over existing methods

**Amortized inference across datasets.** Unlike C-CHVAE or CeFlow, which must be retrained for every new dataset, CounterPFN adapts to a new dataset through in-context learning—a single forward pass. This mirrors TabPFN's key advantage for classification: zero fitting time. A practitioner provides a labeled dataset, a query instance, and an intervention specification; the model returns a counterfactual in milliseconds.

**Ground-truth supervision from SCMs is the critical differentiator.** Existing generative counterfactual methods train on observational data where true counterfactuals are fundamentally unobservable—the "fundamental problem of causal inference." They approximate counterfactual quality through proxy losses (proximity + validity + plausibility). CounterPFN, by contrast, trains on SCM-simulated data where the ground-truth counterfactual is known exactly via the abduction-action-prediction procedure. The model can learn the *actual* mapping from (factual, intervention) → counterfactual, not just an approximation of desirable properties.

**Causal consistency is structural, not regularized.** In methods like Mahajan et al. or CARMA, causal constraints are added as regularization terms to an otherwise causality-agnostic objective. In CounterPFN, causal structure is *built into the training data*—every training counterfactual was generated by a real SCM. The model implicitly learns that intervening on a parent must propagate to descendants (because it sees this in every training example), that intervening on a child should not affect parents (because the SCM never produces such examples), and that colliders create non-intuitive dependencies (because the SCM faithfully implements them). This inductive bias is more robust than post-hoc regularization.

**Inherent support for uncertainty quantification.** Because the PFN framework approximates Bayesian inference, CounterPFN's outputs naturally encode uncertainty. When the context set provides ambiguous information about the causal structure, the model's predicted counterfactual will reflect this uncertainty—potentially outputting a distribution over counterfactual values rather than a point estimate. This is valuable for practitioners who need to know how confident the system is in its counterfactual recommendation.

**Limitations and open challenges.** The synthetic-to-real gap is the primary concern: real-world causal structures may be more complex, involve latent confounders, or follow functional forms not well-represented in the prior. The approach assumes the SCM prior has sufficient coverage—if real-world mechanisms are fundamentally different from what the prior generates, counterfactuals may be unreliable. Scalability constraints inherited from TabPFN (quadratic attention complexity) limit applicability to datasets with tens of thousands of samples and hundreds of features. The quality of the trained classifier used during synthetic data generation affects what "validity" means—a mismatch between the synthetic classifier and the real-world classifier being explained could degrade performance. Finally, evaluating counterfactual quality on real data remains inherently challenging because ground-truth counterfactuals are unobservable.

---

## Conclusion

The proposed CounterPFN architecture occupies a unique niche in the counterfactual explanation landscape. By unifying three independently validated ideas—PFN-style in-context learning (proven effective for tabular classification), SCM-based counterfactual generation (the gold standard for causal counterfactuals), and dual-attention transformers (capable of learning both within-instance causal propagation and cross-instance distributional constraints)—it offers a path to **amortized, causally grounded counterfactual explanations that generalize across datasets without retraining**.

The most novel technical contribution is the training strategy: using SCM-simulated factual–counterfactual pairs as supervised training data for a generative transformer. This sidesteps the fundamental problem of causal inference (unobservable counterfactuals) during training, while the in-context learning mechanism allows the model to adapt to new real-world datasets where true causal structure is unknown. The closest existing works—CausalPFN for treatment effects, TabPFGen for data generation, and Miller and Schölkopf's analysis of in-context counterfactual reasoning—each address a piece of this puzzle, but none combine them into a counterfactual explanation system. The SCM prior's complexity, the curriculum design across training stages, and the multi-objective loss balancing validity against causal consistency represent the critical engineering challenges that will determine whether this theoretical architecture translates to practical counterfactual quality.
