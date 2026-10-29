---
name: scientific-writing
description: Plan, draft, revise, or proofread academic prose — thesis chapters, paper sections, abstracts, captions, cover letters — where claims must trace to evidence, terminology must stay consistent, and structure must follow scholarly convention. Use for research-backed argumentation, citation discipline, claim-scope control (don't let an association become a causal claim), and publication-style wording in English or Persian academic text.
---

# Scientific Writing

## Governing principle

Preserve the author's evidence and intended meaning while making the argument
clear, reproducible, and no stronger than the evidence supports. Never invent a
result, mechanism, citation, number, or claim. When evidence is missing, say so
instead of smoothing over the gap.

## Before drafting: build a writing contract

For any section, paragraph, or claim, pin down:

1. **Function** — what this unit must accomplish for the reader (motivate, report,
   interpret, bound a limitation).
2. **Claim** — the narrowest defensible statement the evidence actually supports.
3. **Evidence** — the exact source: experiment config, dataset/split, metric
   definition, sample count, statistical test, or citation. If you can't name the
   source, don't write the claim yet — go find it.
4. **Scope** — direction, magnitude, uncertainty, and whether the result
   generalizes beyond the tested condition. Distinguish correlation from causation
   explicitly; never let "improves" quietly become "causes."

Draft only once the contract is coherent. Don't polish prose into fluency if its
evidence or placement is wrong — fix the substance first.

## Structure conventions

Follow standard section functions; don't force a fixed template onto content that
doesn't fit it:

- **Introduction**: gap in existing work → research question → contribution,
  in that order. State the contribution precisely enough that a reader could
  falsify it.
- **Methods**: enough detail that someone else could reproduce the result —
  model/config, dataset, preprocessing, evaluation protocol, hyperparameters.
  Name assumptions explicitly; don't bury them in a dependent clause.
- **Results**: lead with the evidence (figure, table, metric), then interpret.
  Never state an interpretation before the reader has seen the number. Report
  negative or null results with the same rigor as positive ones.
- **Discussion/Limitations**: a documented boundary and its consequence, not a
  vague hedge. If something is a fixable bug rather than a genuine limitation,
  fix it — don't launder it into "future work."
- **Conclusion/Abstract**: synthesize only from claims already established in the
  body. The abstract is not where a new, unverified claim first appears.

## Claim and citation discipline

- Every empirical claim needs a traceable source: a specific experiment run, a
  specific cited work, or a derivation shown in the text. "Studies show" without
  a citation is not acceptable.
- Match citation to claim strength — a single pilot study supports "suggests,"
  not "demonstrates" or "proves."
- When paraphrasing a source, preserve its qualifiers (approximately, typically,
  under condition X) — stripping a hedge while keeping the number overstates the
  original claim.
- Flag a hallucination risk explicitly: if asked to add a citation and no real
  source is known, say so rather than fabricating an author/year/venue.

## Terminology and consistency

- Pick one term per concept and hold it everywhere — captions, table headers,
  abstract, body, and conclusion must use the same name for the same thing. If
  this project already has fixed terminology or macros for a concept (check for
  glossary/acronym files and LaTeX commands before introducing a new term),
  reuse them instead of inventing a synonym.
- Cross-check numbers that appear in more than one place (abstract vs. results
  table vs. conclusion). A changed number in one location that isn't propagated
  everywhere is a correctness bug, not a style nit.

## Material-change classification

Before editing, classify the change so the review depth matches the risk:

- **Semantic** — changes the claim, contribution, or causal framing. Re-check the
  entire evidence chain and every downstream summary.
- **Evidence** — changes a number, figure, table, dataset, or method detail.
  Re-check every place that number or figure is referenced.
- **Metadata** — authors, affiliations, funding, declarations. Re-check every
  submission artifact that duplicates it.
- **Surface** — grammar, punctuation, phrasing with no scientific content change.
  Still re-read for meaning drift; a "simplification" that drops a qualifier is a
  semantic change in disguise.

## Four-pass review (for a full section or chapter)

1. **Argument and structure** — purpose, questions, organization, paragraph
   openings, contribution.
2. **Evidence and scope** — methods, results, figures, tables, citations,
   uncertainty, claim strength.
3. **Prose quality** — clarity, natural sentence subjects, necessary vs.
   avoidable repetition, consistent terminology. Run the `academic-humanizer`
   skill here if the text reads as stiff or AI-flavored.
4. **Delivery integrity** — cross-references, numbering, citation list
   completeness, consistent units and decimal conventions.

If a later pass finds a problem, fix it and re-check the passes it affects — don't
certify an earlier pass as still valid.

## Reporting

When reporting the outcome of a writing task, state: what changed, what evidence
backs each changed claim, which passes were run, and any claim you could not
verify (name it rather than silently dropping or guessing it).
