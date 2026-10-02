# POC run findings

The repaired POC improved answers on our ten development questions, but still made material legal omissions or errors on five. This supports continuing the POC; it does not establish production readiness.

## Main comparison

The completed original and repaired POC runs used the same ten questions, model, model digest and sampling settings. These are historical runs, not a fresh benchmark of the current default configuration.

| Measure | Original | Repaired POC |
|---|---:|---:|
| Questions completed | 10/10 | 10/10 |
| Positive answer flag / accepted prose draft | 3/10 | 10/10 |
| Final question errors | 0 | 0 |
| Required-source/status checks passed | Not measured | 9/10 |
| Complete answers with correct references in internal review | 0/10 | 5/10 |

The two answer flags measure different things and are not accuracy scores. The original POSH definition was substantively useful but had an incorrect section reference, so it failed the strict completeness criterion. Five accepted POC drafts also failed that criterion. The quality review was internal, not an independent legal evaluation.

## Where it helped

| Question topic | Original result | POC result |
|---|---|---|
| POSH definition | Useful definition, wrong section label | All five forms with the correct section reference |
| Desertion duration | Abstained | At least two continuous years immediately before the petition |
| Adoption eligibility | Abstained | Suitability, spousal consent, single-applicant rules and regulatory qualification |
| Defective-goods relief | Abstained | Replacement, refund, interest and applicable compensation |
| Consumer complaint limitation | Wrong source and sixty-day passage | Two years, with sufficient cause and recorded reasons for late admission |

## What improved across iterations

- The first broad POC run had five planner errors and only two accepted prose answers. The repaired run had no final question errors and ten accepted drafts; source/status checks improved from 5/10 to 9/10.
- Keeping question parts explicit and constraining source IDs reduced planning and formatting failures. Invalid extracted facts could be rejected without aborting the whole answer.
- A separate POSH regression set improved from 24/30 to 28/30 automated source/status passes. This measures evidence handling, not full answer correctness.
- In one confirmed recovery example, the first retrieval missed the confidentiality provision. Recovery fetched Section 16 and retained its anonymity exception, changing the evidence status from partial to covered.

## Remaining failures

- **POSH filing deadline:** found the route and basic period, but confused the extension conditions and omitted written reasons.
- **Divorce and custody:** omitted the eighteen-month motion limit and overstated a qualified sixty-day disposal endeavour as a mandatory rule.
- **Vehicle transfer:** gave the correct main deadlines but missed out-of-State document requirements.
- **GST registration:** retrieved relevant evidence but omitted threshold qualifications in the final prose.
- **GST payment:** missed scope and shortfall qualifications and mixed Section 74 citations into a Section 73 answer.

The main remaining issue is preserving legal conditions and exceptions when turning retrieved evidence into prose. Better retrieval alone did not solve it.

## Decision and limits

Continue with qualification-preserving answers and citation validation, then evaluate on unseen questions before making an adoption decision. These ten cases were used to develop the POC, so the gains may not generalize. Prepared Act-specific corpora, full-section retrieval and the controller changed together; this comparison cannot attribute the gains to recovery alone. No speed advantage is claimed.

Syntax and controller contract checks passed. Two questions also completed end to end on the current configured model; a full comparison on that configuration was not completed. The POC remains limited to six pinned Acts and does not establish support for rules, schedules, later amendments or multi-Act reasoning.

Detailed answers, prompts, evidence and stepwise traces remain local. This summary is the only committed run-result document.
