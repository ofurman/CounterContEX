---
date: 2026-08-11T13:05:03+02:00
researcher: Oleksii Furman
topic: "NeurIPS 2026 workshop submission practicality and framing for CounterContEX"
tags: [neurips-2026, workshops, counterfactual-explanations, algorithmic-recourse, tabpfn, submission-strategy]
sources: [official-conference-site, openreview, official-workshop-sites]
status: complete
last_updated: 2026-08-11
last_updated_note: "Added follow-up research on generative, representation-learning, reconstruction, and conditional-imputation workshop fit"
---

# Research: NeurIPS 2026 workshop submission practicality for CounterContEX

**Date**: 2026-08-11T13:05:03+02:00
**Researcher**: Oleksii Furman

## Research Question

Research submission practicality for the NeurIPS 2026 workshops most likely to fit CounterContEX, using the official OpenReview group and official workshop sites. Determine whether workshops are accepted/active versus only proposals, likely or announced submission deadlines, page limits, archival/non-archival status, dual-submission rules, required topics, and whether empirical early-stage work is welcome. Also compare possible framing strategies for a zero-shot sparse actionable counterfactual explanation paper built on TabPFN. Return a ranked shortlist with URLs and citations to primary sources; clearly mark missing 2026 details and avoid filling them from prior years without labeling historical evidence.

## Summary

The strongest submission is not the most obvious “interpretability” workshop: CounterContEX’s corrected validity, mislabeled actionability metric, clipped-versus-unclipped LOF, OOB analysis, and full-split stability checks make a particularly strong evaluation-audit paper for TAE. EconML is the strongest recourse/application framing because its 2026 call explicitly names strategic classification and AI decision-making in economic contexts, while HELOC supplies a credit-domain case. EIML is the strongest negative-result/foundation-model framing because it explicitly welcomes works in progress, unsafe extrapolation, blind spots, stress tests, and negative results. Interp4Discovery welcomes negative results and non-language/vision modalities but requires a genuine knowledge-discovery angle; ordinary local recourse alone is not enough. Every ranked venue below has an official subgroup under `NeurIPS.cc/2026/Workshop`, unlike a proposal-only site such as AgentSD; subgroup existence is the clearest current evidence of acceptance into the 2026 workshop program.

## Detailed Findings

### Conference-wide status and dates

- NeurIPS scheduled workshops for Dec 11–12 in Sydney and Dec 12–13 in Paris and Atlanta. Its official call set July 11 as proposal notification, suggested Aug 29 for workshop-paper submission, and required workshop decisions by Sept 29 ([NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops)).
- The live official parent group is [`NeurIPS.cc/2026/Workshop`](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop). A named child subgroup is treated here as accepted/active program evidence; a standalone site that still says “proposal” without such a subgroup is not.

### Ranked shortlist

#### 1. TAE — Can We Trust AI Evaluation? (Sydney)

- **Status:** accepted/active: [official OpenReview subgroup](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FTAE) and [official 2026 CFP](https://tai-eval.github.io/cfp/).
- **Deadline:** Aug 29, 2026 AoE; author notification Sept 22. The site labels these dates “indicative,” so recheck before submission.
- **Length/review:** up to 8 pages, references and appendix excluded; double-blind.
- **Publication/dual:** explicitly non-archival. The 2026 CFP does **not** state a dual-submission rule; do not infer one.
- **Scope:** submissions must study evaluation protocols themselves, rather than only the evaluated model. Explicit topics include robustness to metrics/data splits/seeds, black-box audits, measurement validity, stress tests, finance, and high-stakes applications.
- **Early-stage:** not explicitly invited. Empirical audit work is clearly in scope, but it should be presented as a complete protocol-level study.
- **Best framing:** “When Counterfactual Metrics Lie: A TabPFN Case Study.” Lead with the corrected target-label definition, the library’s mislabeled actionability metric, clipped/unclipped plausibility inversion, OOB reporting, and full-split stress test. Treat the generator as the instrument used to expose evaluation failure modes. This is the best match to the evidence already in the repository.

#### 2. EconML — Economics for Machine Learning (Atlanta)

- **Status:** accepted/active: [official OpenReview subgroup](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FEconML) and [official 2026 workshop site/CFP](https://econml26-workshop.github.io/).
- **Deadline:** Aug 29, 2026 AoE; notification Sept 29; camera-ready date is still TBA.
- **Length/review:** short 4 pages or long 9 pages, references and appendix excluded; double-blind.
- **Publication/dual:** non-archival. Already archival-accepted papers are barred. Multiple NeurIPS-workshop submissions are discouraged. Extended abstracts of work under review elsewhere are allowed if the other venue permits it.
- **Scope:** explicitly includes strategic classification, AI decision-making and bias in economic contexts, and empirical evidence. At least one author must present in person in Atlanta.
- **Early-stage:** emerging topics are encouraged, but the call asks for a rigorous formal model; a 4-page short paper is practical.
- **Best framing:** “Zero-Shot Algorithmic Recourse for Credit Decisions with Tabular Foundation Models.” Define recourse as a strategic intervention problem, make HELOC central, distinguish feasible/actionable interventions from mere feature perturbations, and analyze how sparse suggestions interact with plausibility. The current arbitrary six-feature immutable split needs stronger domain justification; without it, the economics claim is vulnerable.

#### 3. EIML — Epistemic Intelligence in Machine Learning (Paris)

- **Status:** accepted/active: [official OpenReview subgroup](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FEIML) and [official 2026 CFP](https://eiml.cc/calls-for-papers/).
- **Deadline:** Aug 29, 2026; notification Sept 29; camera ready Oct 18.
- **Length/review:** **missing from the 2026 site and live submission form as of Aug 11.** Review/anonymity details are also not stated. Ask the organizers before drafting to a page target.
- **Publication/dual:** **not stated for 2026.** The live OpenReview form requires CC BY 4.0 and says accepted submissions will be public after the conference, but that is insufficient to label the venue archival. Do not import a prior-edition policy.
- **Scope:** unsafe extrapolation, distribution shift, uncertainty representation, blind spots, stress tests, and negative results are explicit topics.
- **Early-stage:** explicitly welcomes works in progress and mature research alike.
- **Best framing:** “Knowing When Zero-Shot Recourse Fails.” Use TabPFN posterior samples/calibration plus the MOONS–HELOC contrast to study epistemic failure under sparse conditioning. The key claim should be a detectable knowledge boundary and an abstention/fallback rule; a generator-only paper with no uncertainty-to-action mechanism would underuse the call.

#### 4. Interpretability for Discovery (Atlanta)

- **Status:** accepted/active: [official OpenReview subgroup](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FInterp4Discovery) and [official 2026 CFP](https://interpretability4discovery.github.io/cfp.html).
- **Deadline:** Aug 29, 2026 AoE; notification Sept 29.
- **Length/review:** requirements are explicitly tentative; currently 5 main-text pages, plus one camera-ready page; references and appendix excluded; double-blind/private review.
- **Publication/dual:** non-archival; work under review at ICLR/NeurIPS is welcome; prior non-archival work is welcome; already archival-accepted work is barred except the workshop’s NeurIPS fast track.
- **Special requirement:** a responsible-use statement is mandatory and omission is grounds for desk rejection.
- **Scope/early-stage:** accepts methodological, empirical, theoretical, and position work; failure cases and negative results are explicitly welcome. It seeks interpretable internal representations that yield non-obvious, testable world knowledge, including beyond language and vision.
- **Best framing:** “Counterfactual Probing of Tabular Foundation Models.” Counterfactuals must become probes that recover and validate conditional regularities learned by TabPFN. Merely producing actionable recourse is a weak match; add a concrete discovery claim and external validation, or prefer TAE/EIML.

#### 5. AI4GOOD — Trustworthy AI for Good (Paris)

- **Status:** accepted/active: [official OpenReview subgroup](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FAI4GOOD) and [official 2026 CFP](https://trustworthy-ai-for-good.github.io/).
- **Deadline:** Aug 29, 2026 AoE; notification Sept 29.
- **Length/review:** 2–9 pages, references and appendix excluded; double-blind.
- **Publication/dual:** non-archival. The FAQ permits simultaneous review elsewhere and even submission to another NeurIPS workshop from AI4GOOD’s side, subject to the other venue’s rules.
- **Scope/early-stage:** model evaluation/auditing/failure modes, evidence standards, accountable public-sector AI, and societal benefit. The multi-agent track expressly welcomes empirical and position papers, but CounterContEX belongs in the general track.
- **Best framing:** “Actionable, Auditable Recourse for High-Stakes Tabular Decisions.” Center who can act on a suggested feature and how harmful recommendations are prevented. HELOC is relevant, but the present evaluation lacks human/domain validation, fairness analysis, or a deployment evidence standard; add one before positioning the paper as AI-for-good.

#### 6. NewInML (Paris) — conditional safety-net venue

- **Status:** accepted/active: [official OpenReview subgroup](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FNewInML) and [official 2026 CFP](https://newinml.github.io/NewInML2026NeurIPS/).
- **Eligibility:** only authors who have not yet published at a top ML conference (NeurIPS, ICML, ICLR, etc.). Verify **all-author** interpretation with organizers if the team is mixed; the page does not clarify it.
- **Deadline:** Aug 29, 2026 AoE; notification Sept 29.
- **Length/review:** 2–8 pages, references excluded; double-blind.
- **Publication/dual:** non-archival; concurrent submission/publication elsewhere is allowed from the workshop’s side.
- **Scope/early-stage:** all ML topics, with a mission focused on helping newcomers polish ideas and experiments. This is the most forgiving method-paper venue if eligible, but it is not a topical community for recourse or tabular foundation models.

### Additional candidates and exclusions

- **MLxOR** is accepted/active ([OpenReview](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FMLxOR), [site](https://mlxor-2026.github.io/)), offers a useful Aug 31 deadline, four pages, non-archival status, and welcomes foundation models, finance, uncertainty-aware decisions, and GenAI+OR. It is a weaker fit unless the sparse greedy procedure is formalized as an optimization/decision method rather than an explanation heuristic.
- **GDDL** is accepted/active and explicitly offers 2–4 pages for early-stage work, 5–9 pages for long work, non-archival/concurrent submission, deadline Aug 29 ([official CFP](https://gddl-neurips-2026.github.io/)). Despite the phrase “structured data,” its technical scope is geometry, optimal transport, graphs/manifolds, and distributional deep learning; ordinary tabular counterfactuals do not fit without a substantive geometry/distribution contribution.
- **Interpretability as a Science** is accepted/active ([OpenReview](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FInterpScience), [CFP](https://interpscience.github.io/cfp)) and non-archival, with 5/9-page tracks and an Aug 28 deadline, but its 2026 scope is specifically LLM interpretability. TabPFN recourse is not a credible match. It also bans simultaneous submission to another workshop.
- **AgentSD** should not be counted as accepted on present evidence: its site still calls itself a “NeurIPS 2026 Workshop Proposal,” gives no accepted venue/date, and has no child subgroup in the official OpenReview workshop group ([site](https://agent-sd.com/)).

## Key Insights

1. **Best paper from existing evidence:** the TAE evaluation-audit framing requires the least new science and turns the project’s corrections into contributions rather than embarrassment.
2. **Best classic recourse paper:** EconML, but only after stronger actionability semantics and domain grounding. A sparse L0 count is not by itself actionable recourse.
3. **Best honest early-stage paper:** EIML, if the authors turn posterior dispersion/OOB into an abstention or uncertainty-aware fallback rule. It explicitly welcomes works in progress and negative evidence.
4. **Do not oversell TabPFN novelty.** The contribution should be the zero-training conditional-generation/recourse mechanism, sparse search, or evaluation insight—not merely applying a foundation model.
5. **The HELOC failure is useful evidence.** TAE and EIML reward careful failure analysis; hiding it would weaken the most venue-specific framing.

## Confidence Notes

- EIML’s page limit, review mode, archival label, and dual-submission rule are genuinely absent from the current 2026 official pages. The CC BY field in OpenReview is not enough to infer archival status.
- TAE says its dates are “indicative,” although the site and live venue are active.
- Interp4Discovery calls its five-page format provisional; recheck before final formatting.
- Workshop sites can change until submissions close. The official OpenReview child group is the most reliable acceptance-status signal; the workshop CFP is usually the richest policy source.

## Open Questions

1. Is the author team eligible for NewInML (no author has a top-ML-conference publication), and does at least one author commit to the required Atlanta presentation for EconML?
2. Can the team add either (a) an evaluation protocol comparison for TAE, (b) a domain-grounded actionability validation for EconML/AI4GOOD, or (c) an uncertainty-triggered abstention rule for EIML before the deadline?
3. Will the paper be concurrently submitted anywhere else? That choice materially changes the viable venues, especially EconML and InterpScience.

## Clarifications Log

N/A

## Follow-up Research 2026-08-11 13:18

### Question

Do accepted NeurIPS 2026 workshops on autoencoders, representation learning, latent-variable/generative models, diffusion/flow, reconstruction/imputation, or conditional generation offer a better fit if CounterContEX is reframed as an autoencoder-like generator or as TabPFN conditional imputation?

### Finding

**TabPFN conditional imputation materially improves fit to PriGM; “autoencoder-like generator” does not.** CounterContEX has no learned encoder, bottleneck latent code, decoder, or reconstruction objective. Its actual mechanism is autoregressive conditional imputation by a pretrained tabular foundation model, with target-label conditioning and inference-time sparse search. Calling that an autoencoder would be technically inaccurate and would not satisfy the geometry/representation requirements of NeurReps or GDDL. The accurate generative framing is conditional sampling under partial observation and extreme masking, including the empirical limits of in-context adaptation under distribution shift.

### Ranked generative/representation shortlist

#### 1. PriGM — Principles of Generative Modeling (Paris): genuinely stronger method fit

- **Accepted/active:** [official OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FPriGM) and [official CFP](https://sites.google.com/view/prigmneurips2026/call-for-papers).
- **Exact scope:** principled mathematical, physical, or rigorous empirical analysis of generative modeling. The 2026 call explicitly asks about model classes and expressivity, in-context learning, inference-time computation/adaptation, data structure, and distribution shift.
- **Deadline/format:** Sept 5, 2026 AoE; four single-column pages plus unlimited references and appendices; double-blind.
- **Publication/dual:** non-archival; under-review and ongoing unpublished work is welcome; already archival-accepted work is not.
- **Fit:** strong only if the paper is reframed from “a new recourse tool” to a principled empirical study of when a pretrained tabular model can act as a conditional generator without retraining. The MOONS/HELOC contrast, feature-ordering ablation, mask dimensionality, posterior calibration, OOB behavior, and context selection can answer a workshop-level question about the limits of inference-time conditional generation.
- **Mismatch/risk:** PriGM is theory-oriented. A two-dataset application paper with no general principle, controlled scaling law, or mechanism-level explanation will look thin. Add masking-ratio/context-size/feature-order sweeps and state a falsifiable conclusion about conditional-generation failure.

#### 2. GDDL — Geometric Distributional Deep Learning (Paris): conditional, not automatically improved

- **Accepted/active:** [official OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FGDDL) and [official CFP](https://gddl-neurips-2026.github.io/).
- **Exact scope:** systems that jointly model geometry and distributions; geometry-aware optimal transport, distributions on graphs/manifolds, scalable structured-space methods, and generative models on non-Euclidean domains.
- **Deadline/format:** Aug 29 AoE; 2–4 pages for early-stage work or 5–9 pages for long papers; double-blind.
- **Publication/dual:** non-archival; concurrent/subsequent submission explicitly allowed.
- **Fit:** at most moderate if CounterContEX adds a real distributional-geometry component—e.g., manifold-support constraints, OT/geodesic proximity, or a geometric analysis explaining why sparse conditional samples leave the data manifold.
- **Mismatch:** “structured data” in this CFP means geometric/distributional structure, not ordinary tables. TabPFN conditional imputation or an autoencoder analogy alone does not meet scope.

#### 3. NeurReps — Symmetry and Geometry in Neural Representations (Sydney): poor without new representation analysis

- **Accepted/active:** [official workshop/CFP](https://neurreps.org/) and three official OpenReview tracks: [Proceedings](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FNeurReps_Proceedings), [Extended Abstracts](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FNeurReps_Extended_Abstracts), and [Findings](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FNeurReps_Findings).
- **Exact scope:** symmetry, geometry, topology, invariant/equivariant representations, representational geometry, and geometric/topological analysis of generative models.
- **Deadline/format:** Aug 24 AoE. Proceedings: 9 pages, archival, double-blind. Extended abstract: 4 pages, non-archival, double-blind, explicitly suitable for early-stage or negative findings. Findings: no page limit, single-blind, but aimed at high-impact experimentalist–theorist work.
- **Dual:** Proceedings requires at least 30% new/unsubmitted material and imposes a similar 30% extension requirement for later publication; Extended Abstracts have no restrictions.
- **Fit:** weak. An actual study of TabPFN latent geometry, permutation/feature-order symmetries, or manifold structure could fit; reconstruction/imputation performance does not by itself study neural representations.

#### 4. AIDaR — AI Data Readiness for Scientific Discovery (Paris): imputation keyword, wrong task/domain

- **Accepted/active:** [official OpenReview venue](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FAIDaR) and [official CFP](https://aidar-workshop.github.io/2026/).
- **Scope:** scientific-data organization, data systems, benchmarks/evaluations, workflow/tool demos, and failure analyses tied to downstream scientific model or agent behavior.
- **Deadline/format:** Aug 29 AoE; 4-page short/work-in-progress or 8-page full; double-blind; non-archival.
- **Fit:** poor unless the work is redirected to missing-value/data-readiness infrastructure on a real scientific dataset. Generating credit recourse from HELOC is not scientific data readiness.

### Explicit exclusions

- **DiffuLM** and **BeNTo** are about diffusion/flow *language models* and next-generation decoding, not tabular conditional generation.
- **GDDL** covers diffusion/flows only on geometric structured spaces; it is not a generic generative-model workshop.
- **World Models in Physical AI**, PTA, and Continual World Models require embodied/sequential decision-making or learned environment dynamics. A static tabular counterfactual is not a world model.
- **ATTRIB** mentions conditional generation only for textual/data attribution and provenance; this does not match counterfactual recourse.
- No accepted 2026 workshop discovered has a general autoencoder, reconstruction, missing-data imputation, or tabular-representation-learning CFP.

### Recommended framing choice

1. For the **existing evidence with minimal additional experiments**, keep the TAE evaluation-audit framing from the main report.
2. For a **generative-model paper**, target PriGM and use the accurate phrase **“zero-shot conditional generation by masked autoregressive imputation with a tabular foundation model.”** Center inference-time adaptation limits, not the application label “counterfactual explanation.”
3. For a **recourse paper**, keep EconML/AI4GOOD and treat conditional imputation as the mechanism, not the headline.
4. Do **not** use “autoencoder-like” unless a real encoder–decoder/reconstruction model is implemented and compared. The label creates expectations the current method cannot meet and does not unlock a better workshop.

## Sources Consulted

- [NeurIPS 2026 Call for Workshops](https://neurips.cc/Conferences/2026/CallForWorkshops) — official program-wide dates and workshop purpose.
- [NeurIPS 2026 OpenReview workshop parent](https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop) — authoritative accepted/active workshop subgroup hierarchy.
- [TAE 2026 CFP](https://tai-eval.github.io/cfp/) — scope and submission policy.
- [EconML 2026 CFP](https://econml26-workshop.github.io/) — scope, tracks, dual policy, attendance requirement.
- [EIML 2026 CFP](https://eiml.cc/calls-for-papers/) — scope, dates, and works-in-progress policy.
- [Interpretability for Discovery 2026 CFP](https://interpretability4discovery.github.io/cfp.html) and [scope](https://interpretability4discovery.github.io/about.html) — provisional format, negative-results policy, and discovery criterion.
- [AI4GOOD 2026 CFP](https://trustworthy-ai-for-good.github.io/) — general-track scope, short-paper practicality, and dual-submission FAQ.
- [NewInML 2026 CFP](https://newinml.github.io/NewInML2026NeurIPS/) — eligibility and non-archival policy.
