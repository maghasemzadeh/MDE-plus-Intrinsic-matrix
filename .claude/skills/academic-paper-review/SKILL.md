---
name: academic-paper-review
description: Review a thesis, paper, or chapter for readiness — weaknesses, novelty, rigor, and revision priorities — rather than making a specific requested edit. Use when the user asks "is this ready," "what's weak here," "review this chapter," "what would a reviewer say," or wants a prioritized list of issues before deciding what to fix. For executing a specific known edit, use `scientific-writing` instead; this skill is diagnosis, not editing.
---

# Academic Paper Review

This skill produces a reviewer-style assessment. It diagnoses; it does not edit.
If the user wants something fixed immediately, say what you'd flag and then route
to `scientific-writing` (prose/structure) or `academic-humanizer` (AI-tell removal)
for the actual edit.

## Review passes

Run each pass top to bottom over the full scope requested (chapter, section, or
whole document). Don't skip to line-editing before finishing the structural pass.

### 1. Contribution and novelty

- What is the single claimed contribution? If you can't state it in one sentence
  after reading the introduction, that's the first finding.
- Is the contribution actually new relative to what's cited as prior work, or is
  it an incremental variant presented as novel? Check whether the related-work
  section honestly positions the gap, or overstates how unexplored it is.
- Does every major claim in the abstract/conclusion appear, with evidence, in the
  body? A claim that exists only in the abstract is unsupported.

### 2. Rigor and evidence

- For every empirical claim: is there a named dataset/split, sample count,
  metric definition, and comparison baseline? Flag claims resting on a single
  run with no variance/uncertainty reported.
- Check the baseline comparison is fair: same data split, same evaluation
  protocol, same preprocessing. A result that looks like an improvement but
  used a different crop/split/checkpoint than the baseline is a correctness bug,
  not a minor caveat — flag it as high severity.
- Check statistical claims ("significant improvement," "consistently better")
  are backed by an actual test or enough repeated trials, not a single number
  comparison.
- Distinguish correlation from causation in every claim that uses causal
  language ("improves," "causes," "leads to").

### 3. Structure and completeness

- Does each chapter/section serve a clear function, and does the ordering
  follow the field's convention (motivation → method → evidence → discussion)?
- Are limitations stated as specific, falsifiable boundaries ("tested only on
  indoor scenes with max depth 10m") rather than vague hedging ("further work is
  needed")?
- Are there unresolved `\todo`s, placeholder figures, or `[citation needed]`
  markers remaining? List them explicitly — don't let the prose quality of
  surrounding text hide an incomplete section.

### 4. Internal consistency

- Do numbers reported in the abstract, results tables, and conclusion match
  exactly? Cross-check every number that appears more than once.
- Is terminology consistent (same name for the same concept throughout)?
- Do all figure/table/equation references resolve, and does every figure have
  a caption that stands alone without requiring the body text to interpret it?

## Severity classification

Report findings tagged by severity so the user can triage:

- **Blocking** — a claim is unsupported, a comparison is unfair/invalid, or a
  number is inconsistent across the document. Must fix before this is
  submission- or defense-ready.
- **Major** — missing baseline, weak statistical support, unclear contribution
  statement. Should fix.
- **Minor** — prose quality, formatting, non-critical clarity issues. Can defer.

## Reporting format

Lead with a one-paragraph verdict: ready / ready with minor fixes / not ready,
and why. Then list findings grouped by severity, each with:
- the specific location (chapter/section/page or line if available)
- what's wrong, stated concretely (not "could be clearer" — say what's missing)
- what evidence or change would resolve it

Never give a blanket "looks good" without having run all four passes. If scope
was limited (e.g., only one chapter reviewed), say so explicitly rather than
implying the whole document was checked.
