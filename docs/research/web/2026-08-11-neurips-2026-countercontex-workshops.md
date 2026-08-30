---
date: 2026-08-11T13:04:47+02:00
researcher: Oleksii Furman
topic: "Official NeurIPS 2026 workshops relevant to CounterContEX"
tags: [neurips-2026, workshops, counterfactual-explanations, tabular-ml, tabpfn, interpretability]
sources: [official-conference-pages, openreview, official-workshop-sites]
status: complete
last_updated: 2026-08-11
last_updated_note: "Added comprehensive CFP audit for XAI, counterfactual explanation, recourse, and responsible-interpretability scope"
---

# Research: NeurIPS 2026 workshops relevant to CounterContEX

**Date**: 2026-08-11T13:04:47+02:00
**Researcher**: Oleksii Furman

## Research Question

Research the official NeurIPS 2026 workshop list at https://openreview.net/group?id=NeurIPS.cc/2026/Workshop and enumerate workshops plausibly relevant to CounterContEX: zero-shot sparse actionable counterfactual explanations for tabular classifiers, using pretrained TabPFN as a conditional density estimator, target-label conditioning, greedy one-feature-at-a-time generation, and no retraining.

## Summary

The best topical candidate is Interpretability for Discovery, but its call requires a credible knowledge-discovery framing. ML×OR is a strong alternative if CounterContEX is framed as generative decision support or algorithmic recourse. AI4GOOD is the broadest trustworthy-AI fallback. GDDL, Trust-AI-Eval, and PriGM are conditional fits that require, respectively, a geometric/distributional structured-data contribution, an evaluation-protocol contribution, or a contribution to generative-model principles. Two workshops whose titles look especially relevant—Interpretability as a Science and XAI4Science—have narrow LLM-only and weather/climate-only calls.

## Detailed Findings

### Status of the OpenReview group

- NeurIPS directed workshop proposals to the separate `NeurIPS.cc/2026/Workshop_Proposals` venue and scheduled proposal acceptance notifications for July 11, 2026 ([NeurIPS Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops), [proposal portal](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop_Proposals)).
- The requested `NeurIPS.cc/2026/Workshop` group contains configured workshop venues and submission workflows, not proposal submissions ([OpenReview group](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop), [official OpenReview API](https://api2.openreview.net/groups?parent=NeurIPS.cc%2F2026%2FWorkshop&limit=1000)). This is strong evidence that the entries are accepted workshops. The page does not itself visibly label the collection an accepted-workshop list, and its children include auxiliary track groups, so a raw child count is not a workshop count.

### 1. Interpretability for Discovery

- Exact title: **NeurIPS 2026 Workshop on Interpretability for Discovery** ([official site](https://interpretability4discovery.github.io/), [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/Interp4Discovery)).
- Scope: methodological, empirical, theoretical, and position work using interpretability to uncover new knowledge; topics include methods, evaluation frameworks, and model designs for discovery across unfamiliar architectures and modalities, as well as failure cases and negative results ([scope](https://interpretability4discovery.github.io/about.html), [CFP](https://interpretability4discovery.github.io/cfp.html)).
- Fit: strongest interpretability match, but CounterContEX needs a credible domain-discovery story rather than only explaining classification decisions.
- Details: submission deadline August 29, 2026, 23:59 AoE; notification September 29; workshop December 12 or 13 in Atlanta, with the final date to be confirmed. Requirements are expressly provisional: up to five main-text pages, one additional camera-ready page, double-blind, non-archival, and private during review ([CFP](https://interpretability4discovery.github.io/cfp.html)).

### 2. ML×OR

- Exact title: **NeurIPS 2026 Second Workshop on ML×OR: Mathematical Foundations and Operational Integration of Machine Learning for Uncertainty-Aware Decision-Making** ([official site and CFP](https://mlxor-2026.github.io/), [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/MLxOR)).
- Scope: new methodology, theory, and applications at the ML–OR intersection, with an emphasis on GenAI+OR; the call explicitly includes data-driven optimization and operational decision-making, foundation models, causal inference, distributional robustness, and simulation ([official CFP](https://mlxor-2026.github.io/)).
- Fit: strong if sparse feature changes are framed as actionable recourse, inference-time generative decision support, or constrained optimization.
- Details: maximum four pages, non-anonymous, non-archival; deadline August 31 AoE; notification September 29 AoE; workshop December 12 or 13 in Atlanta ([official CFP](https://mlxor-2026.github.io/)).

### 3. AI4GOOD

- Exact title: **Trustworthy AI for Good (AI4GOOD) Workshop @ NeurIPS 2026** ([official site and CFP](https://trustworthy-ai-for-good.github.io/), [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/AI4GOOD)).
- Scope: trustworthy models, including evaluation, auditing, harmful failure modes, alignment, and robustness, plus methods and evidence standards for real-world social benefit ([official CFP](https://trustworthy-ai-for-good.github.io/)).
- Fit: plausible when actionable counterfactual explanations are tied to accountability, fairness, high-stakes decisions, or concrete social benefit.
- Details: two to nine pages, double-blind, non-archival; deadline August 29 AoE; notification September 29; camera-ready November 29; workshop December 12 ([official CFP](https://trustworthy-ai-for-good.github.io/)).

### 4. Geometric Distributional Deep Learning

- Exact title: **NeurIPS Workshop: Bridging Optimal Transport, Learning and Structured Data: Toward Geometric Distributional Learning** ([official site and CFP](https://gddl-neurips-2026.github.io/), [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/GDDL)).
- Scope: combining geometry and distributions to build scalable, efficient, interpretable models for structured data; topics include geometry-aware optimal transport, distributional architectures on non-Euclidean domains, scalable structured-space methods, and generative models on graphs and manifolds ([official CFP](https://gddl-neurips-2026.github.io/)).
- Fit: conditional. CounterContEX matches distribution modeling, interpretability, and structured/tabular data, but lacks an explicit geometry or optimal-transport contribution.
- Details: short papers of two to four pages or long papers of five to nine pages; double-blind and non-archival; deadline August 29 AoE; notification September 29; final program October 16; workshop December 12–13 ([official CFP](https://gddl-neurips-2026.github.io/)).

### 5. Trust-AI-Eval

- Exact title: **NeurIPS 2026 Trust-AI-Eval Workshop: Can We Trust AI Evaluation?** ([official CFP](https://tai-eval.github.io/cfp/), [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/TAE)).
- Scope: evaluation protocols themselves, rather than only evaluated models; topics include black-box auditing, measurement and causal validity, uncertainty and robustness, and deployment-risk or decision-aware metrics ([official CFP](https://tai-eval.github.io/cfp/)).
- Fit: plausible only if CounterContEX makes a central contribution to evaluating the sparsity, actionability, plausibility, faithfulness, or robustness of counterfactual explanations.
- Details: up to eight pages, double-blind, non-archival; submission opens July 30; deadline August 29 AoE; review deadline September 14; notification September 22; final program September 27; workshop December 11 or 12 in Sydney ([official CFP](https://tai-eval.github.io/cfp/)).

### 6. Principles of Generative Modeling

- Exact title: **NeurIPS 2026 Workshop on Principles of Generative Modeling (PriGM)** ([official CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers), [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PriGM)).
- Scope: principled understanding of generative modeling, including expressivity, in-context capabilities, inference-time computation and adaptation, distribution shift, and the properties of pretrained models that enable post-training ([official CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers)).
- Fit: conditional. Using pretrained TabPFN as a zero-shot conditional density model and performing inference without retraining is relevant, but an application method without a new insight into generative-model principles is likely too weak.
- Details: four single-column pages, anonymized, non-archival; deadline September 5 AoE; review September 6–22; notification September 29 ([official CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers)).

### Misleading titles / poor fit

- **Interpretability as a Science: NeurIPS 2026 Workshop** is specifically about scientific foundations of *LLM* interpretability, including measurement, causality, and falsifiability. A tabular-classifier counterfactual method is outside the stated scope unless it becomes an LLM-interpretability paper. The deadline is August 28 AoE; notification September 29; camera-ready November 15; short papers up to five pages and long papers up to nine; non-archival; workshop December 11 or 12 in Sydney ([official CFP](https://interpscience.github.io/cfp), [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/InterpScience)).
- **XAI4Science: Knowledge Discovery and Trust through Interpretable Foundation Models** focuses specifically on weather and climate foundation models. Its threads are ante-hoc interpretability, post-hoc attribution and evaluation, and physics-consistent explanations. Without a weather or climate application, CounterContEX is a poor fit. Regular papers are up to eight pages and tiny papers up to five; anonymity is optional; the venue is non-archival; deadline August 29 AoE; notification September 28; workshop December 11 or 12 in Sydney ([official site and CFP](https://xai4science.github.io/), [OpenReview](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/XAI4Science)).

## Sources Consulted

- [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops) — proposal portal, dates, and acceptance timeline.
- [NeurIPS 2026 Workshops Guidance](https://neurips.cc/Conferences/2026/WorkshopsGuidance) — official workshop contribution and notification rules.
- [OpenReview NeurIPS 2026 Workshop group](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop) — requested venue collection.
- The official workshop sites linked in each finding — scope, calls, formats, and dates.

## Key Insights

CounterContEX does not have a perfect workshop match in the current collection. The most credible strategies are to emphasize interpretability supporting domain discovery for Interp4Discovery, or sparse actionable recourse and inference-time decision support for ML×OR. Broad trustworthy-AI positioning can work for AI4GOOD but benefits from a concrete high-stakes or social-good use case.

## Confidence Notes

The configured `Workshop` children are strongly consistent with accepted venues because proposals used a different official portal and the acceptance date has passed. However, the OpenReview group page does not itself provide a plain-text statement that every child is an accepted workshop, and no separate official accepted-workshop index was found.

## Open Questions

- Does the CounterContEX paper include a scientific-discovery or high-stakes social-good application that can support a workshop-specific framing?
- Is evaluation methodology itself a novel contribution, or is the evaluation section conventional benchmarking of a new method?

## Clarifications Log

N/A

## Follow-up Research 2026-08-11 13:15

### Comprehensive XAI and recourse audit

The official OpenReview directory contains 105 apparent main workshop groups after excluding eight obvious auxiliary track groups ([official OpenReview API](https://api2.openreview.net/groups?parent=NeurIPS.cc%2F2026%2FWorkshop&limit=1000)). A scan of every linked official workshop site, followed by manual review of every relevant CFP hit, found no workshop that explicitly solicits general counterfactual explanations, algorithmic recourse, or actionable explanations for tabular classifiers.

The honest ranking is:

1. **Interpretability for Discovery** is the sole direct interpretability-method option, but requires novel, testable knowledge discovery rather than ordinary decision explanation ([scope](https://interpretability4discovery.github.io/about.html), [CFP](https://interpretability4discovery.github.io/cfp.html)).
2. **ML×OR** can accommodate CounterContEX as target-conditioned, inference-time generative decision support or recourse, but a substantive OR or operational-decision formulation is needed ([official CFP](https://mlxor-2026.github.io/)).
3. **AI4GOOD** can accommodate responsible, accountable use of explanations in a concrete high-stakes or social-good application, but its CFP does not explicitly name XAI or recourse ([official CFP](https://trustworthy-ai-for-good.github.io/)).
4. **Geometric Distributional Deep Learning** matches structured data, distributions, and interpretability, but expects geometry or optimal transport ([official CFP](https://gddl-neurips-2026.github.io/)).
5. **Economics for Machine Learning** is plausible only for recourse in economic decisions such as lending, hiring, or resource allocation; it does not explicitly solicit explanations or recourse ([official CFP](https://econml26-workshop.github.io/)).
6. **Trust-AI-Eval** fits only if evaluation of counterfactual explanations is itself the methodological contribution ([official CFP](https://tai-eval.github.io/cfp/)).

Prominent title or keyword matches are not general CounterContEX venues: XAI4Science is restricted to weather and climate foundation models ([official CFP](https://xai4science.github.io/)); Interpretability as a Science is restricted to LLM interpretability foundations ([official CFP](https://interpscience.github.io/cfp)); ATTRIB concerns training-data and output provenance rather than feature-level counterfactual explanations ([official CFP](https://attrib-workshop.cc/)); and IAB concerns agent trajectories and human-agent interaction ([official site](https://iab-agents.github.io/)). Other CFP uses of “counterfactual” concern physical reasoning, clinical treatment outcomes, or LLM social simulation rather than counterfactual explanations or recourse ([Physical Understanding for Decision-Making](https://sites.google.com/view/neurips-2026-workshop-pudm), [World Models for High-Stakes Health](https://wmhs-neurips.github.io/WMHS/), [SocialAgent](https://social-llm-workshop.github.io/)).
