---
name: paragraph-editor
description: Revise Persian thesis paragraphs for fluency and clarity while preserving the original meaning, structure, evidence, numbers, and scientific claims. Use for paragraph-by-paragraph thesis editing in LaTeX.
---

# Paragraph editor

Use this skill when revising an existing Persian thesis paragraph. The goal is to make the paragraph fluent, clear, and easy to understand while preserving what it says. Do not turn paragraph editing into summarization, fact checking, or rewriting the research argument.

## Core requirements

- Preserve the paragraph's purpose, logical role, scientific meaning, evidence, limitations, and conclusion.
- Do not delete, invent, weaken, strengthen, reorder, or reinterpret claims unless the user explicitly requests that change.
- Preserve every number, metric, dataset, model, test, confidence interval, citation, equation, and technical condition unless the user explicitly authorizes a change.
- Keep the overall content and scope intact. The normal edit is local wording improvement, not removal of material.
- Use the surrounding paragraphs and tables to resolve references and pronouns. If the intended meaning remains unclear, report the ambiguity instead of guessing.
- Keep technical terms consistent throughout the thesis. Change terminology only when the user requests it or when the existing term is demonstrably misleading; report such a terminology change.

## Persian prose style

- Write in plain, fluent Persian with direct sentence structure.
- Split long sentences when they contain several comparisons, qualifications, or results.
- End every sentence with a complete predicate, preferably an explicit verb. Do not end a sentence with a bare number, parenthesis, table fragment, or unfinished phrase.
- Explain every number by stating what it measures and which model, dataset, metric, ratio, interval, or test it belongs to.
- Do not repeat a table row in prose. Summarize the finding and refer to the table for the detailed values.
- State the complete point and stop. Avoid padding, repetition, exaggerated explanation, and material that is not needed to understand the paragraph’s purpose.
- Keep table captions short. Put methodological explanations in the surrounding text.
- Avoid ambiguous statistical wording. Explain the test, threshold, and interpretation in plain language when needed.

## Minimal editing workflow

1. Read the target paragraph together with the preceding and following paragraphs, related table captions, and referenced labels.
2. Identify the paragraph's purpose and list its claims, values, comparisons, and limitations before editing.
3. Rewrite only what is needed for clarity, fluency, grammar, sentence length, and terminology consistency.
4. Compare the revised paragraph against the original claim by claim. Confirm that no scientific content, number, caveat, or conclusion changed.
5. If the paragraph is in LaTeX, preserve commands, labels, references, citations, equations, and review markup.
6. Run the Codex pass (section below) on the revised paragraph and apply the accepted findings.
7. Build the thesis when the source is changed, and report any remaining warnings or errors.

## Codex pass (mandatory)

After making the edit and before building, run the Codex CLI over the changed
paragraph, as described in "Codex pass for Persian prose edits" in
`.claude/skills/thesis-editor/SKILL.md`. This applies to every paragraph revision,
summary, merge or split of Persian prose:

```bash
cd thesis && codex exec --sandbox read-only --skip-git-repo-check "<prompt>"
```

Point Codex at the file and the paragraph (label or opening words), give it the
original wording or the tables/sections to compare against, and ask for unsupported
or changed claims, Persian grammar/spelling/half-space problems, and unclear
sentences with a suggested rewrite. Codex only reviews; apply the accepted findings
yourself, keep the thesis macros and terminology, and never accept a suggestion that
changes a claim or number. Report what Codex changed, or that it could not be run.

## Review markup

For an existing paragraph, mark only the words or phrases that changed. Keep unchanged text outside review macros. Use `\\edrep{old}{new}` for replacements, `\\edins{...}` for insertions, and `\\edrep{old}{}` for removals. Do not wrap an entire existing paragraph in a yellow, green, or red environment merely because part of it changed. A whole paragraph may be green only when it is entirely new; a whole deleted paragraph may be red only when it is intentionally removed.

## Output

Report:

- the revised paragraph;
- the meaning and claims preserved;
- any terminology or interpretation issue that required user attention;
- what the Codex pass changed (or that Codex could not be run);
- the validation or build result when LaTeX was edited.
