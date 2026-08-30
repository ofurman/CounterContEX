---
date: 2026-08-11T13:04:29+02:00
researcher: Oleksii Furman
topic: "NeurIPS 2026 workshops that best fit CounterContEX"
tags: [neurips-2026, workshops, counterfactual-explanations, recourse, tabular-foundation-models, interpretability]
sources: [official-openreview, official-workshop-sites]
status: complete
last_updated: 2026-08-11
last_updated_note: "Added follow-up research on tabular and foundation-model technical fit"
---

# Research: NeurIPS 2026 workshop fit for CounterContEX

**Date**: 2026-08-11T13:04:29+02:00
**Researcher**: Oleksii Furman

## Research Question

Independently identify and rank NeurIPS 2026 workshops from the official OpenReview group that best fit CounterContEX: a research idea for zero-shot counterfactual explanations for tabular classifiers using pretrained TabPFN as a conditional density estimator, with target-class conditioning, greedy sparse/actionable feature changes, offline inference, no retraining. Focus on thematic fit (XAI, recourse/counterfactuals, trustworthy/responsible ML, causal reasoning, tabular foundation models/generative modeling). Return top candidates with exact official titles/URLs, direct evidence from calls/scopes, reasoning, and weaknesses. Cite every factual claim from official sources; note uncertainty.

## Summary

No NeurIPS 2026 workshop visible in the official OpenReview venue hierarchy is explicitly dedicated to counterfactual explanations, algorithmic recourse, or tabular foundation models; this is an inference from the official workshop titles and the closest calls, not a claim made by NeurIPS ([official NeurIPS OpenReview directory](https://openreview.net/venue?id=NeurIPS.cc)). The strongest thematic match is **Interpretability as a Science**, because its call explicitly includes criteria for genuine explanation, formal interpretability, causal/interventional methods, and evaluation validity, though its framing is LLM-centric ([official call](https://interpscience.github.io/cfp)). **Trust-AI-Eval** is next if CounterContEX foregrounds black-box auditing and rigorous evaluation, while **Economics for Machine Learning** and **ML×OR** become plausible if the paper is framed respectively as strategic classification/recourse or constrained decision optimization. Workshops centered on generative modeling, geometric distributional learning, scientific discovery, or data attribution are real but weaker fits because CounterContEX uses those ideas instrumentally rather than advancing their central research questions.

## Detailed Findings

### Ranked candidates

#### 1. Interpretability as a Science: NeurIPS 2026 Workshop — strongest overall (high, with a scope caveat)

- Official OpenReview group: [Interpretability as a Science: NeurIPS 2026 Workshop](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FInterpScience).
- Direct scope evidence: the call welcomes work on “criteria for genuine explanation,” formal and mathematical frameworks for interpretability, “causal and interventional methods,” measurement validity, identifiability, evaluation design, and falsifiability ([official CFP](https://interpscience.github.io/cfp)).
- Fit inference: CounterContEX can be presented as a formal post-hoc explanation method for black-box tabular models, with sparsity/actionability/plausibility criteria and controlled evaluation. The conditional density component can support a precise plausibility claim, while target-class conditioning and feature interventions support an interventional framing.
- Weakness: the workshop describes itself as pursuing the scientific foundations of **LLM interpretability** ([official CFP](https://interpscience.github.io/cfp)); TabPFN is not a language model, and counterfactual recourse does not by itself establish causal effects. The paper should avoid equating feasible feature changes with structural causal interventions unless it adds a causal model or defensible assumptions.

#### 2. NeurIPS 2026 Trust-AI-Eval Workshop — strong only if evaluation/auditing is central (medium-high)

- Official OpenReview group: [NeurIPS 2026 Trust-AI-Eval Workshop](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FTAE).
- Direct scope evidence: the workshop treats evaluation itself as an object of study and explicitly solicits black-box auditing, measurement and causal validity, robustness/stress tests, application-domain evaluation, and analysis of when offline metrics justify real-world deployment decisions ([official workshop scope](https://tai-eval.github.io/)).
- Fit inference: CounterContEX is naturally compatible with black-box behavioral auditing and offline evaluation. A strong submission would make counterfactual evaluation—not only generation—a core contribution: validity, sparsity, actionability, plausibility, stability, runtime, and failure modes across tabular tasks and classifiers.
- Weakness: the call is about trustworthy **evaluation**, not primarily about proposing a new explanation generator ([official workshop scope](https://tai-eval.github.io/)). A method paper with ordinary benchmark tables would be weaker than a paper built around auditing claims or a rigorous evaluation protocol.

#### 3. Workshop on Economics for Machine Learning at NeurIPS 2026 — compelling recourse/strategic framing (medium)

- Official OpenReview group: [Workshop on Economics for Machine Learning at NeurIPS 2026](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FEconML).
- Direct scope evidence: its call explicitly lists strategic classification, AI decision making and bias in economic contexts, discrete choice and behavioral modeling, and incentive/information-gap models ([official workshop page and CFP](https://econml26-workshop.github.io/)).
- Fit inference: actionable counterfactual recourse concerns how an affected person can change features to obtain a desired model outcome, which can be positioned at the strategic-classification boundary. Immutable/actionable constraints and costs can be modeled as feasible choices rather than generic perturbation norms.
- Weakness: the present CounterContEX description contains no incentives, strategic response, welfare, institutional decision context, or economic model. The fit depends on adding an explicit recourse/strategic-classification question; merely calling feature edits “actions” is unlikely to be enough.

#### 4. NeurIPS 2026 Second Workshop on ML×OR: Mathematical Foundations and Operational Integration of Machine Learning for Uncertainty-Aware Decision-Making — decision/optimization route (medium)

- Official OpenReview group: [NeurIPS 2026 Second Workshop on ML×OR: Mathematical Foundations and Operational Integration of Machine Learning for Uncertainty-Aware Decision-Making](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FMLxOR).
- Direct scope evidence: the call welcomes methodological work at the ML/OR intersection, including decision-focused generative models, generative AI in data-driven optimization, and work broadly relevant to decision-making through foundation models, causal inference, simulation, distributional robustness, and optimization ([official CFP](https://mlxor-2026.github.io/)).
- Fit inference: sparse actionable recourse can be formulated as constrained optimization balancing class attainment, change cost, feasibility, and density-based plausibility. The pretrained TabPFN component supplies the foundation-model/generative-estimation angle, and the greedy procedure supplies a decision algorithm.
- Weakness: a heuristic greedy feature search without OR structure, guarantees, or serious comparison to optimization baselines may look peripheral. The workshop emphasizes GenAI+OR and uncertainty-aware operational decisions, whereas CounterContEX currently describes a post-hoc XAI method.

#### 5. NeurIPS 2026 Workshop on Principles of Generative Modeling (PriGM) — method-component fit, contribution mismatch (medium-low)

- Official OpenReview group: [NeurIPS 2026 Workshop on Principles of Generative Modeling (PriGM)](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FPriGM).
- Direct scope evidence: the call asks for principled understanding of generative modeling, including model classes and distributions, in-context learning/generalization, inference-time computation and adaptation, and which pretrained-model properties enable post-training ([official CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers)).
- Fit inference: zero-shot use of a pretrained TabPFN as a target-conditioned density estimator, followed by inference-time search with no retraining, overlaps with inference-time computation and pretrained-model behavior.
- Weakness: CounterContEX uses conditional density estimation as a component but does not primarily advance the scientific understanding or theory of generative models. Unless the paper studies why TabPFN's conditional distributions support valid counterfactual search, the match is superficial.

#### 6. NeurIPS 2026 Workshop: Bridging Optimal Transport, Learning and Structured Data: Toward Geometric Distributional Learning — structured-distribution angle, but weak core match (low-medium)

- Official OpenReview group: [NeurIPS 2026 Workshop: Bridging Optimal Transport, Learning and Structured Data: Toward Geometric Distributional Learning](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FGDDL).
- Direct scope evidence: the workshop asks how geometry and distributions can yield scalable, efficient, interpretable models for structured data and lists generative models, but its detailed topics focus on optimal transport, graphs, manifolds, and non-Euclidean domains ([official CFP](https://gddl-neurips-2026.github.io/)).
- Fit inference: tabular observations are structured and CounterContEX relies on conditional probability distributions and sparse moves, so there is a broad conceptual overlap.
- Weakness: the current method has no optimal transport, graph/manifold geometry, or non-Euclidean structured-data contribution. Ordinary tabular structure alone is unlikely to satisfy the workshop's geometric center of gravity.

#### 7. NeurIPS 2026 Workshop on Interpretability for Discovery — XAI label, wrong purpose (low-medium)

- Official OpenReview group: [NeurIPS 2026 Workshop on Interpretability for Discovery](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FInterp4Discovery).
- Direct scope evidence: the workshop welcomes interpretability methods and evaluation frameworks across unfamiliar architectures and modalities, but specifically frames them as tools for uncovering novel, testable knowledge about the world ([official scope](https://interpretability4discovery.github.io/about.html), [official CFP](https://interpretability4discovery.github.io/cfp.html)).
- Fit inference: CounterContEX is unquestionably an interpretability method and TabPFN is an unusual architecture/modality relative to language and vision.
- Weakness: actionable counterfactual explanations explain or contest predictions; they do not necessarily discover new scientific knowledge. The official page explicitly distinguishes discovery from “just explanation” ([official scope](https://interpretability4discovery.github.io/about.html)).

#### 8. XAI4Science: Knowledge Discovery and Trust through Interpretable Foundation Models — only with a weather/climate application (low for the generic paper)

- Official OpenReview group: [XAI4Science: Knowledge Discovery and Trust through Interpretable Foundation Models](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FXAI4Science).
- Direct scope evidence: the call includes post-hoc attribution/evaluation and physics-consistent explanations, but the workshop is organized around trustworthy weather and climate foundation models ([official workshop and CFP](https://xai4science.github.io/)).
- Fit inference: foundation-model-based XAI is topically adjacent.
- Weakness: without a weather/climate use case or scientific-discovery contribution, generic tabular classifier recourse is outside the workshop's domain-specific focus.

### False friends and other near-misses

- [Third NeurIPS Workshop on Attributing Model Behavior at Scale: Data Attribution and Provenance](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FATTRIB) is about tracing model outputs/actions to **training data**, with contributive and corroborative attribution, provenance, and downstream accountability—not feature attribution or counterfactual explanations ([official CFP](https://attrib-workshop.cc/)). It is not a good target unless CounterContEX is substantially reframed around training-data influence.
- [Trustworthy AI for Good (AI4GOOD) Workshop @ NeurIPS 2026](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FAI4GOOD) sounds broad, but its current CFP lists multi-agent security topics such as collusion, delegation, confinement, propagated attacks, red-teaming agent populations, and multi-agent oversight ([official CFP](https://trustworthy-ai-for-good.github.io/)). Generic tabular recourse is a poor match.
- [Neurips 2026 Workshop - LIGHT: Deployable Small Foundation Models](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FLIGHT) includes explainability, trustworthiness, and deployment, but centers distillation, compression, quantization, pruning, and compact foundation-model systems ([official scope](https://almaai-disi-unibo.github.io/neurips2026-light-smallModels/)). CounterContEX's no-retraining/offline character is adjacent, but it does not compress or design a compact model.

## Sources Consulted

- [Official NeurIPS OpenReview venue directory](https://openreview.net/venue?id=NeurIPS.cc) — authoritative workshop-group inventory.
- [Interpretability as a Science CFP](https://interpscience.github.io/cfp) — explanation, intervention, and evaluation scope.
- [Trust-AI-Eval](https://tai-eval.github.io/) — auditing and evaluation scope.
- [Economics for Machine Learning](https://econml26-workshop.github.io/) — strategic classification and decision-making scope.
- [ML×OR 2026](https://mlxor-2026.github.io/) — decision optimization, causal inference, and foundation-model scope.
- [PriGM CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers) — generative-model principles and inference-time computation.
- [GDDL CFP](https://gddl-neurips-2026.github.io/) — distributional/geometric structured-data scope.
- [Interpretability for Discovery scope](https://interpretability4discovery.github.io/about.html) and [CFP](https://interpretability4discovery.github.io/cfp.html) — interpretability-for-scientific-discovery boundary.
- [XAI4Science](https://xai4science.github.io/) — weather/climate foundation-model XAI scope.
- [ATTRIB](https://attrib-workshop.cc/), [AI4GOOD](https://trustworthy-ai-for-good.github.io/), and [LIGHT](https://almaai-disi-unibo.github.io/neurips2026-light-smallModels/) — near-miss checks.

## Key Insights

The best submission strategy is determined more by the paper's claimed contribution than by its components. If the novelty is the explanation method and its validity, target **Interpretability as a Science**. If the novelty is a rigorous counterfactual audit/evaluation framework, target **Trust-AI-Eval**. If the paper formalizes affected individuals' feasible actions and strategic response, **Economics for Machine Learning** is credible. If it formalizes recourse as constrained prescriptive optimization with meaningful algorithmic analysis, **ML×OR** becomes credible. The mere presence of TabPFN or conditional density estimation is not enough to make PriGM or GDDL the best home.

## Confidence Notes

- Confidence is high in the exact titles and venue identities because they come from the official OpenReview workshop groups.
- Confidence is medium in the rank ordering because workshop fit depends on the eventual paper's experiments and framing, which are not fully specified.
- Workshop pages and calls may still change before their submission deadlines. The ranking reflects pages available on 2026-08-11.
- “No exact-fit workshop” is an inference from the reviewed official inventory and calls; a workshop may use broader unpublished reviewer interpretations than its public wording suggests.

## Open Questions

None required to answer the current ranking request. The ranking should be revisited if CounterContEX adds a causal structural model, strategic-agent analysis, or a domain-specific weather/climate application.

## Clarifications Log

No clarifications requested.

## Follow-up Research 2026-08-11 13:18

### Question

Do any accepted NeurIPS 2026 workshops directly cover tabular machine learning, tabular foundation models, pretrained/in-context learning, structured data, conditional density estimation, or foundation-model inference-time methods suitable for CounterContEX?

### Result: no dedicated tabular or conditional-density workshop

The official NeurIPS 2026 OpenReview workshop hierarchy contains no workshop whose official title or reviewed CFP is dedicated to tabular ML, tabular foundation models, TabPFN, or conditional density estimation ([official NeurIPS OpenReview directory](https://openreview.net/venue?id=NeurIPS.cc)). Searches for those terms surfaced papers and non-accepted/proposal pages, but no dedicated accepted workshop group. Therefore there is no exact technical match; the candidates below are fallbacks whose public scopes cover only parts of CounterContEX.

### Technical-fallback ranking

1. **[NeurIPS 2026 Workshop on Principles of Generative Modeling (PriGM)](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FPriGM)** — best foundation-model fallback. Its CFP explicitly covers classes of distributions, in-context learning, inference-time computation/adaptation, and properties of pretrained models ([official CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers)). CounterContEX directly uses a pretrained in-context tabular model as a conditional estimator and performs zero-shot inference-time search. **Mismatch:** PriGM asks for principled understanding of generative modeling; an XAI application that merely consumes TabPFN is peripheral. Format: four single-column pages plus unlimited references/appendices; deadline September 5, 2026 AoE; non-archival ([official CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers)).

2. **[PTA: From Pretrained Representations to Acting Agents](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FPTA)** — strongest literal match to zero-shot/test-time action search. The call names test-time search and optimization under objectives/constraints, and zero-/few-shot decision-making from pretrained representations ([official CFP](https://ptaworkshop.github.io/call-for-papers.html)). **Mismatch:** its center is sequential control, RL, robotics, planning, and acting agents; tabular counterfactual edits are not automatically a sequential agent problem. Format: short up to 4 pages or full up to 9 pages; deadline August 29, 2026 AoE; non-archival ([official CFP](https://ptaworkshop.github.io/call-for-papers.html)).

3. **[AXIOM: Foundations of Efficient Deep Learning](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FAXIOM)** — plausible if efficiency is a measured contribution. The CFP includes adaptive inference, test-time adaptation, memory-efficient inference, efficient foundation models, and interpretability of efficient models ([official CFP](https://axiom-neurips2026.github.io/)). **Mismatch:** offline/no-retraining operation is not itself a contribution to efficient deep learning; the paper would need resource/runtime analysis and efficient-inference baselines. Format: 4-page papers or 1-page Grand Challenges submissions; deadline August 29, 2026 ([official CFP](https://axiom-neurips2026.github.io/)).

4. **[On-Device Intelligence: Foundation Models under Real-World Constraints](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FODI)** — exact overlap with offline local execution. Its CFP covers efficient inference and reasoning, reliable local execution in offline/weak-connectivity settings, and deployment benchmarks measuring latency, energy, and memory ([official CFP](https://odi2026.github.io/)). **Mismatch:** CounterContEX does not currently target mobile/edge hardware or resource budgets. Format: up to 5 pages, double-blind and non-archival; deadline August 29, 2026 AoE ([official CFP](https://odi2026.github.io/)).

5. **[Neurips 2026 Workshop - LIGHT: Deployable Small Foundation Models](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FLIGHT)** — adjacent through explainability and deployment. It invites explainability/reasoning/verification in small models, efficient adaptation beyond fine-tuning, runtime guarantees, and deployment reports ([official CFP](https://almaai-disi-unibo.github.io/neurips2026-light-smallModels/cfp.html)). **Mismatch:** the workshop centers small/compressed foundation models, distillation, quantization, and pruning; CounterContEX neither compresses TabPFN nor establishes that model compactness is its contribution. The site lists non-archival submissions and an August 29, 2026 deadline, but does not state a page limit ([official CFP](https://almaai-disi-unibo.github.io/neurips2026-light-smallModels/cfp.html), [dates](https://almaai-disi-unibo.github.io/neurips2026-light-smallModels/dates.html)).

6. **[NeurIPS 2026 Workshop: Bridging Optimal Transport, Learning and Structured Data: Toward Geometric Distributional Learning](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FGDDL)** — closest title-level match for distributions and structured data. It seeks distributional learning, generative models, and interpretable models for structured data ([official CFP](https://gddl-neurips-2026.github.io/)). **Mismatch:** its actual scope centers optimal transport, graphs, manifolds, and non-Euclidean data; generic tabular conditional density estimation has no clear geometric contribution. Format: 2–4-page short or 5–9-page long papers; deadline August 29, 2026 AoE; non-archival ([official CFP](https://gddl-neurips-2026.github.io/)).

### Other inspected near-misses

- **[Epistemic Intelligence in Machine Learning](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FEIML)** covers probability representations, epistemic versus aleatoric uncertainty, misspecification, and uncertainty-guided decisions ([official scope](https://epistemic-intelligence-in-ml.github.io/)). A density-based plausibility score is not necessarily epistemic uncertainty, so this is weak without an uncertainty contribution. Deadline: August 29, 2026; workshop site does not expose a paper-length rule.
- **[Transitioning from Pre-Training to Post-Training](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FPre-to-Post)** studies LLM training-stage interactions and explicitly focuses fine-tuning, RL, distillation, and training dynamics ([official CFP](https://pretrain2posttrain.github.io/call.html)). CounterContEX deliberately performs no post-training, so “uses a pretrained model” is insufficient. Short papers are 4–5 pages; deadline August 29, 2026 AoE.
- **[Neural Network Artifacts as a New Data Modality](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FNeuralArtifacts)** includes interpretability and model selection without retraining, but studies populations of weights, gradients, representations, and training traces as data ([official scope](https://artifactsasdata.org/)). CounterContEX queries one pretrained model rather than learning from neural artifacts.
- **[Foundation Models for Temporal Systems: From Forecasting to World Modeling](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FFMTS)** covers time-series foundation models, generative temporal modeling, intervention, and calibrated probabilistic forecasts ([official scope](https://fmts-workshop.github.io/)). It is relevant only if CounterContEX is redesigned for temporal data.

### Recommendation

Keep **Interpretability as a Science** as the best overall thematic venue. If a technical foundation-model venue is specifically desired, **PriGM is the best fallback**, but only if the manuscript elevates TabPFN's conditional/in-context behavior into a studied contribution rather than treating it as a replaceable scoring component. If the paper instead formalizes recourse as constrained test-time action search, **PTA** or **ML×OR** becomes more defensible, with the sequential-agent or OR mismatch stated explicitly.
