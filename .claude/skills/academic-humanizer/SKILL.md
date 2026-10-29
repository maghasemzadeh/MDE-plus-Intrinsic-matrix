---
name: academic-humanizer
description: Remove AI-sounding patterns from academic/scientific prose (thesis chapters, paper sections, abstracts) while preserving every claim, number, citation, unit, and uncertainty qualifier exactly. Use when text reads as stiff, inflated, or mechanically uniform and needs to sound like a researcher wrote it — not when the request is a general-purpose rewrite (use the broader `humanizer` skill for non-scientific text) and not when the request is about claim correctness or structure (use `scientific-writing` for that).
---

# Academic Humanizer

Scope: prose-level rewriting only. This skill changes *how* a sentence sounds,
never *what* it claims. If a rewrite would change a number, drop a hedge, add
an inference, or alter claim scope, stop and flag it as a `scientific-writing`
concern instead of silently "smoothing" it.

## Invariants — never touch these while humanizing

- Every number, unit, percentage, and statistical value, exactly as given.
- Every citation and its attached claim.
- Every qualifier: "approximately," "under condition X," "in the sampled
  subset," "suggests" vs. "demonstrates." Removing a hedge to make a sentence
  punchier silently overstates the claim — don't do it.
- Technical terms that are the field's standard name for a concept. Don't
  swap a precise term for a more "natural-sounding" synonym if it introduces
  ambiguity (e.g. don't vary between "absolute relative error" and "relative
  depth error" if they denote the same fixed metric name).

## Patterns to remove

**Bureaucratic/stock phrasing** — delete or replace, don't just reword around it:
- "It is worth noting that...", "It should be mentioned that..." → state the
  point directly.
- "plays a significant/crucial role in" → name the actual mechanism.
- "in order to" → "to". "due to the fact that" → "because".
- "a wide range of" / "a variety of" when a specific, countable list is
  available → give the list.

**AI-tell structures:**
- Rule-of-three triads used as filler ("fast, reliable, and efficient") where
  the source doesn't actually support three independent claims.
- "Not only... but also..." where a plain sentence says the same thing.
- Tacked-on explainer clauses: "...which demonstrates/highlights/underscores
  the importance of..." appended reflexively to a sentence that already made
  its point.
- Generic throat-clearing openers: "In today's rapidly evolving field of...",
  "With the rapid advancement of...".
- Formulaic closers: "In conclusion, it can be said that...", "Overall, these
  findings suggest..." when the conclusion section already carries this
  function — don't repeat it at every paragraph's end.
- Uniform sentence rhythm: every sentence 18-22 words, every paragraph the
  same shape. Real academic writing varies — a short declarative sentence
  after a long analytical one is normal, not sloppy.
- Overuse of em dashes as a filler connector where a period or comma reads
  more naturally.

## What stays — don't over-correct

- Formal register and hedged claims are *correct* for academic writing, not an
  AI tell. "These results suggest..." is appropriate scientific caution, not
  something to "humanize" into "These results show...".
- Passive voice is acceptable and often preferred in methods sections
  ("samples were collected") — don't force active voice where the field
  convention is passive.
- Repetition of a technical term across sentences is often *required* for
  precision (see terminology consistency in `scientific-writing`) — don't
  break it up with synonym rotation just to vary wording; that's the opposite
  failure mode, not a fix.

## Procedure

1. Read the passage once for content — list every claim, number, and citation
   it contains. This is the invariant set you must preserve.
2. Rewrite sentence-by-sentence, checking after each sentence that the
   invariant set is still intact.
3. Read the result aloud (or simulate a spoken reading). If a sentence would
   sound unnatural said by a researcher presenting their own work, revise its
   syntax or structure — not its content.
4. Diff the rewrite against the invariant list from step 1. Any number,
   citation, or qualifier that changed is a bug — fix it before returning the
   text.

## Reporting

State explicitly: what stylistic patterns were removed, and confirm the
invariant set (claims/numbers/citations/qualifiers) is unchanged. If a
sentence could not be de-AI-ified without touching a claim, leave it and say
why, rather than silently weakening the science to make the prose sound better.
