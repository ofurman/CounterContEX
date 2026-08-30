---
date: 2026-08-11T13:03:09+02:00
researcher: Oleksii Furman
topic: "Most suitable NeurIPS 2026 workshop for CounterContEX"
tags: [neurips-2026, workshops, counterfactual-explanations, algorithmic-recourse, tabpfn, tabular-ml]
sources: [official-neurips, openreview, official-workshop-sites]
status: complete
last_updated: 2026-08-11
last_updated_note: "Added follow-up research on XAI, tabular foundation models, and autoencoder/generative-model framing"
---

# Research: Most suitable NeurIPS 2026 workshop for CounterContEX

**Date**: 2026-08-11T13:03:09+02:00
**Researcher**: Oleksii Furman

## Research Question

Search for the most suitable NIPS workshop for our idea: https://openreview.net/group?id=NeurIPS.cc/2026/Workshop

## Summary

The best target for CounterContEX as a method paper is the **Workshop on Economics for Machine Learning (EconML)**. Its call explicitly includes strategic classification and AI decision-making and bias in economic settings, which is a closer match to actionable counterfactual recourse—especially with HELOC as the main application—than the superficially attractive interpretability workshops. The recommended paper framing is **zero-shot algorithmic recourse for high-stakes tabular decisions**, with the greedy TabPFN generator cast as a constrained intervention method rather than only a post-hoc explanation. The submission deadline is **August 29, 2026 AoE**; the workshop offers 4-page and 9-page tracks, is double-blind and non-archival, and requires an in-person presenter in Atlanta ([official EconML call](https://econml26-workshop.github.io/)).

If the paper instead makes evaluation failures its central contribution, **Trust-AI-Eval (TAE)** may be an even stronger fit. **AI4GOOD** is the broad responsible-AI fallback, while **NewInML** is a practical fallback only if the team meets its newcomer eligibility rule.

## Detailed Findings

### 1. Recommended: Economics for Machine Learning (EconML)

- The official call explicitly lists **strategic classification**, AI decision-making and bias in economic contexts, discrete-choice and behavioral modeling, and emerging topics grounded in a rigorous formal model ([EconML CFP](https://econml26-workshop.github.io/), [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FEconML)).
- CounterContEX naturally becomes an algorithmic-recourse paper: given a classifier and an affected individual, find a sparse, feasible set of feature changes that obtains the desired decision while remaining plausible under the data distribution.
- HELOC supplies an economically meaningful credit-decision setting. MOONS should remain a diagnostic illustration, not the headline experiment.
- The main gap is that the current project treats actionability largely as a feature mask. A strong EconML submission should justify immutable, mutable, directional, and cost constraints in domain terms and distinguish actionable advice from merely plausible perturbations.
- Practical details: deadline **August 29, 2026 AoE**; short papers up to 4 pages or long papers up to 9 pages, excluding references and appendices; double-blind; non-archival. Multiple NeurIPS-workshop submissions are discouraged, and at least one author must present in Atlanta ([official EconML call](https://econml26-workshop.github.io/)).

Suggested title: **“CounterContEX: Zero-Shot Algorithmic Recourse for Tabular Classifiers with Foundation-Model Priors.”**

Suggested formal core:

\[
\min_{x'} \; \lambda_0\lVert x'-x\rVert_0 + \lambda_c C(x,x') - \lambda_p\log p_{\mathrm{TabPFN}}(x'\mid y_{\mathrm{target}})
\]

subject to the target decision, immutable-feature equality, directional/action constraints, and valid feature domains. The implementation can remain greedy, but the paper should clearly connect each selection and stopping rule to this objective.

### 2. Strong alternative: Trust-AI-Eval (TAE)

- TAE explicitly asks authors to study evaluation protocols themselves, including black-box auditing, measurement and causal validity, stress testing, finance, and whether offline metrics justify deployment claims ([TAE CFP](https://tai-eval.github.io/cfp/), [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FTAE)).
- This is a strong match if the paper leads with the repository's substantive evaluation findings: clipped versus unclipped plausibility, out-of-bound generation, actionability semantics, dense versus sparse conditioning, full-split stability, and failure analysis on HELOC.
- It is a weaker home for a straightforward new-generator paper. The contribution would need to become a counterfactual evaluation or audit protocol, with CounterContEX serving as the principal case study.
- Practical details: indicative deadline **August 29, 2026 AoE**; up to 8 pages; double-blind; non-archival ([TAE CFP](https://tai-eval.github.io/cfp/)).

### 3. Broad responsible-AI alternative: AI4GOOD

- The general track covers trustworthy-model evaluation, auditing and failure modes, evidence standards for real-world social benefit, and accountable AI use ([AI4GOOD CFP](https://trustworthy-ai-for-good.github.io/), [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FAI4GOOD)).
- CounterContEX fits if framed around safe, understandable recourse for consequential credit decisions. It would benefit from domain validation, subgroup/fairness analysis, or evidence that proposed actions are genuinely available to affected people.
- Practical details: deadline **August 29, 2026 AoE**; 2–9 pages; double-blind; non-archival ([AI4GOOD CFP](https://trustworthy-ai-for-good.github.io/)).

### 4. Conditional alternatives

- **EIML** is suitable if the central result becomes knowing when zero-shot recourse fails: unsafe extrapolation, epistemic blind spots, posterior uncertainty, and an abstention/fallback policy. Its call explicitly welcomes works in progress and negative results, with a deadline of August 29, but current format and publication-policy details are incomplete ([EIML CFP](https://epistemic-intelligence-in-ml.github.io/calls-for-papers/), [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FEIML)).
- **NewInML** accepts all ML topics and offers 2–8 page, double-blind, non-archival submissions by August 29. It is available only to researchers who have not yet published at a top ML conference; the public page does not clarify mixed-author-team eligibility ([NewInML CFP](https://newinml.github.io/NewInML2026NeurIPS/), [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FNewInML)).
- **ML×OR** becomes credible if the greedy method is strengthened into a constrained optimization or decision method with analysis and suitable OR baselines. It allows 4-page non-archival submissions until August 31 ([ML×OR CFP](https://mlxor-2026.github.io/), [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FMLxOR)).

### Attractive titles that are poor matches

- **Interpretability as a Science** is explicitly centered on LLM interpretability. A TabPFN recourse method is outside its stated center unless the workshop confirms broader model coverage ([official CFP](https://interpscience.github.io/cfp)).
- **Interpretability for Discovery** seeks interpretability that produces novel, externally testable knowledge about the world, rather than ordinary prediction explanations or recourse ([official scope](https://interpretability4discovery.github.io/about.html), [CFP](https://interpretability4discovery.github.io/cfp.html)).
- **XAI4Science** is specifically about interpretable weather and climate foundation models in 2026 ([official workshop site](https://xai4science.github.io/)).
- **ATTRIB** concerns training-data attribution and provenance, not counterfactual or feature-level explanations ([official workshop site](https://attrib-workshop.cc/)).

## Sources Consulted

- [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops) — conference-wide workshop schedule and proposal process.
- [Official NeurIPS 2026 OpenReview workshop directory](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop) — active workshop venue hierarchy.
- [EconML](https://econml26-workshop.github.io/) — scope and submission policy.
- [Trust-AI-Eval](https://tai-eval.github.io/cfp/) — evaluation/audit scope and submission policy.
- [AI4GOOD](https://trustworthy-ai-for-good.github.io/) — trustworthy-AI general track and submission policy.
- [EIML](https://epistemic-intelligence-in-ml.github.io/calls-for-papers/) — epistemic-failure and negative-result scope.
- [NewInML](https://newinml.github.io/NewInML2026NeurIPS/) — eligibility and submission policy.
- [ML×OR](https://mlxor-2026.github.io/) — optimization/decision scope and submission policy.
- [Interpretability as a Science](https://interpscience.github.io/cfp), [Interpretability for Discovery](https://interpretability4discovery.github.io/cfp.html), [XAI4Science](https://xai4science.github.io/), and [ATTRIB](https://attrib-workshop.cc/) — false-positive scope checks.

## Key Insights

The workshop choice should follow the paper's central claim, not the words “counterfactual,” “foundation model,” or “interpretability” in isolation. CounterContEX has three viable papers inside it: an **algorithmic-recourse method** for EconML, an **evaluation/audit paper** for TAE, or an **epistemic failure and abstention paper** for EIML. The current core idea aligns most naturally with the first. The HELOC failure analysis should remain visible: it can motivate feasibility constraints, uncertainty reporting, and safe refusal rather than appearing only as a negative result.

## Conflicting Information

Thematically, TAE may require less new experimentation because the repository already contains unusually detailed evaluation corrections and failure analysis. However, this would change the paper from the user's stated generation idea into an evaluation paper. EconML is therefore ranked first for fidelity to the idea, while TAE is the strongest route if the team prefers to optimize around the evidence already available.

## Confidence Notes

- Confidence is high that the listed venues are active: they have official child groups under the NeurIPS 2026 OpenReview workshop hierarchy. The separate proposal portal and July proposal-notification date are documented by NeurIPS ([official call](https://neurips.cc/Conferences/2026/CallForWorkshops)).
- Confidence is medium in the final ranking because acceptance fit depends on how much domain-grounded actionability, optimization, or evaluation work can be added before submission.
- Workshop pages may still change. TAE labels its dates indicative, and EIML does not yet state several format and publication details.

## Open Questions

1. Does the team want the paper's primary contribution to be the recourse method, the evaluation audit, or epistemic failure detection?
2. Can at least one author attend the Atlanta workshop if EconML accepts the paper?
3. Is the author team eligible for NewInML, and will the work be concurrently submitted elsewhere?

## Clarifications Log

N/A

## Follow-up Research 2026-08-11 13:30

### Question

Are there better-matched workshops related specifically to XAI, tabular foundation models, or autoencoders?

### Result

There is **no accepted NeurIPS 2026 workshop dedicated to general XAI/counterfactual explanations, tabular ML, TabPFN/tabular foundation models, or autoencoders** in the official workshop hierarchy ([OpenReview directory](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop)). The closest technically honest alternative is **Principles of Generative Modeling (PriGM)**, provided the paper makes TabPFN's conditional-generation behavior—not only the recourse application—a central research contribution.

### XAI options

- **Interpretability for Discovery** is the closest workshop that welcomes general interpretability methods across unfamiliar architectures and modalities. However, its goal is to extract novel, externally testable knowledge about the world. Counterfactual recourse alone does not meet that requirement ([scope](https://interpretability4discovery.github.io/about.html), [CFP](https://interpretability4discovery.github.io/cfp.html)).
- **Interpretability as a Science** covers formal explanation criteria, causal/interventional methods, measurement validity, and falsifiability, but its 2026 call is explicitly about LLM interpretability. Submission would be risky without written confirmation from the organizers that tabular foundation models are welcome ([official CFP](https://interpscience.github.io/cfp)).
- **XAI4Science** is genuinely about XAI and foundation models, but the 2026 edition is restricted to weather and climate models ([official site](https://xai4science.github.io/)).
- Consequently, there is no honest exact-fit general-XAI venue. For the existing method, AI4GOOD, EconML, or ML×OR remains safer than forcing it into one of these calls.

### Tabular foundation-model option: PriGM

PriGM explicitly asks about model classes and distributions, in-context learning, inductive bias, inference-time computation and adaptation, distribution shift, and properties of pretrained models ([official CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers), [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FPriGM)). CounterContEX overlaps through:

- zero-shot conditional density estimation with a pretrained TabPFN;
- target-label conditioning through an appended column;
- masked autoregressive imputation;
- additional inference-time greedy computation without retraining;
- observed failure under severe masking and distribution shift.

The paper would need to ask a principled question such as: **When can in-context tabular foundation models serve as conditional generators, and how do conditioning density, masking, and inference-time search affect validity and support?** A paper that only introduces a counterfactual application is likely too instrumental for PriGM.

Practical details: **September 5, 2026 AoE** deadline; four single-column pages plus unlimited references and appendices; double-blind; non-archival; ongoing and concurrently reviewed work is permitted ([PriGM CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers)). This later deadline and compact empirical format make it a realistic target.

### Why not call it an autoencoder?

CounterContEX is not an autoencoder method: it has no learned encoder, latent bottleneck, decoder, or reconstruction-training objective. TabPFN performs conditional prediction/imputation in context. The accurate description is:

> **Zero-shot conditional generation by masked autoregressive imputation with a tabular foundation model.**

No generic autoencoder/reconstruction workshop was found. **GDDL** and **NeurReps** mention structured distributions or neural representations, but their centers are respectively optimal transport/non-Euclidean geometry and symmetry/representation geometry. CounterContEX has neither contribution today ([GDDL CFP](https://gddl-neurips-2026.github.io/), [NeurReps CFP](https://neurreps.org/)).

### Revised recommendation

| Intended paper | Best target | Required emphasis |
|---|---|---|
| Counterfactual recourse method | EconML or ML×OR | Feasible actions, costs, constrained decision optimization |
| TabPFN/foundation-model paper | **PriGM** | Conditional generation, in-context behavior, inference-time computation, failure limits |
| Evaluation/failure-analysis paper | TAE | Metric validity, OOB behavior, stress tests, deployment claims |
| Responsible-XAI application | AI4GOOD | High-stakes impact, accountability, fairness/domain validation |
| General XAI paper | No exact 2026 workshop | Interp4Discovery only with a real discovery contribution |

Given the requested technical direction, **PriGM is now the recommended venue**, and the paper should avoid autoencoder terminology.
