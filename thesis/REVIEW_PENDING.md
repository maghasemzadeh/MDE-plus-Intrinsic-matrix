# Pending thesis review updates

Colour legend: green = added, yellow = new words in an edited paragraph,
red highlight (black text, no strike) = removed words and deleted parts,
gray = moved without word changes.
Find every region of an update with `grep -n 'REVIEW R<n>' thesis/tex/*.tex`.

## Finalized — 2026-11-18 — all earlier updates (branch `refactor/final`)

At the author's request, every review mark that was still pending was made final before the last
full read of the thesis: R21, R25 and R26 in chapter 5, R30, R31 and R32 in chapter 3 and the English
abstract, and the gray R7 `edmoved` wrapper of §5.2.2. New text kept, deleted text and the old
`tab:hypothesis_verdicts_old` table dropped, `% REVIEW` lines removed. The source text before these
updates is in git history (R26: 1f2833a; R30–R32: b9c7246). The macros in `tex/commands.tex` stay.
