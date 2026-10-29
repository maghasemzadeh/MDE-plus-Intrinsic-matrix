---
name: persian-writing
description: Write and typeset natural, correctly-formatted Persian (Farsi) text — prose register, orthography, and right-to-left behavior inside XeLaTeX/xepersian documents. Use for ANY task touching Persian text in this repo, especially the thesis (thesis/tex/*.tex): writing or revising Persian sentences, fixing نیم‌فاصله/digit/punctuation mistakes, or getting RTL/LTR direction right around mixed Persian-Latin content, math, figures, and captions. Trigger on Persian/Arabic script, یا کلماتی مثل فارسی, پایان‌نامه, راست‌چین, نیم‌فاصله, ایزافه, زی‌پرشین, xepersian.
---

# Persian Writing (XeLaTeX / xepersian)

Two independent failure modes make Persian LaTeX output look wrong, and both
need fixing:

1. **The prose sounds like a translation or like AI**, not like a Persian
   researcher wrote it.
2. **The direction/typography is broken** — Latin digits breaking RTL flow, a
   paragraph that starts with a Latin word and flips direction, wrong ezafe
   kasre, inconsistent spacing.

This project's thesis uses **XeLaTeX + xepersian** (see
`thesis/tex/commands.tex`), not Word/PowerPoint — every rule below is scoped to
that toolchain. For general LaTeX build/compile concerns (not Persian-specific),
use `latex-clean-build`; for prose argument/claim quality, use
`scientific-writing`.

## Register — decide before writing a word

A register mismatch is the costliest mistake here. This is a **thesis**
(پایان‌نامه), so default register is **academic Persian (نگارش علمی)**:

- Measured, hedged claims; passive voice is acceptable and often expected.
- Zero تعارف (no flattering filler), zero colloquial particles (no میشه/می‌خوام/رو
  forms — academic register uses full written forms: می‌شود نه میشه).
- است نه می‌باشد — "می‌باشد" is a classic bureaucratic tell; plain "است" is
  correct and more natural.
- Hold one consistent point of view. State facts impersonally; use «ما» only
  for what the author/research actually did, and don't slide between
  impersonal and first-person within one section.

If a request is for something other than the thesis body (an email, a README,
a commit message in Persian), ask or infer the right register instead of
defaulting to academic — but for anything under `thesis/`, academic register is
the default unless told otherwise.

## Banned bureaucratic and AI tells (Persian)

Remove these on sight, same spirit as `academic-humanizer` but for Persian:

- می‌باشد، لازم به ذکر است، در راستای، از اهمیت ویژه‌ای برخوردار است، نقش
  بسزایی ایفا می‌کند — all stock bureaucratic padding; state the point plainly.
- نه تنها ... بلکه as filler where a plain sentence says the same thing.
- Tacked-on «که نشان‌دهنده‌ی ... است» appended reflexively after a sentence
  that already made its point.
- Generic openers «در دنیای امروز...» and closers «در نهایت می‌توان گفت...».
- No em dashes (—) in Persian prose — restructure or use «،» instead.
- Flat, uniform sentence rhythm (every sentence the same length/shape) reads as
  machine-generated. Real academic Persian still varies sentence length.

The native test: would an Iranian reader screenshot this sentence as "متن هوش
مصنوعی"? If yes, rewrite before delivering — but never by removing a hedge,
qualifier, or number; that's a `scientific-writing` violation, not a style fix.

## Orthography — non-negotiable, every deliverable

1. **ZWNJ (نیم‌فاصله, U+200C)**: می‌شود نه می شود؛ کتاب‌ها نه کتاب ها؛
   بزرگ‌تر، خانه‌ام، به‌عنوان.
2. **Persian letterforms only**: ی (U+06CC) not ي، ک (U+06A9) not ك.
3. **Persian digits ۰۱۲۳۴۵۶۷۸۹** inside Persian prose. Latin digits stay in
   code, file paths, version numbers, and URLs. This thesis already handles
   decimal numbers via `\fadecimal{}` (see below) — use it, don't hand-roll
   digit direction fixes.
4. **Persian punctuation**: ، ؛ ؟ and «گیومه» for quotes, not "straight quotes".
5. **Ezafe kasre**: never write the ezafe as a bare «ـه» — «کتابِ من» not
   «کتابه من». Test by substituting «است»: if the sentence breaks when you
   mentally insert «است», the kasre is required and dropping it is an error
   a native reader will read as careless.
6. **Never start a Persian sentence, line, caption, or table cell with a Latin
   word/term.** Direction is often inferred from the first character; a line
   like "Depth Anything مدل پایه است" can render left-aligned with punctuation
   on the wrong side. Lead with a Persian word: «مدل پایه \lr{Depth Anything
   V2}…» — and see the thesis's own `\basemodel` macro below, which already
   does exactly this.
7. **Never letter-space Persian text** (breaks letter joining) and never
   synthesize bold/italic on a font that doesn't define those weights.

## XePersian / LaTeX direction rules — specific to this thesis

This thesis already defines the macros and conventions below in
`thesis/tex/commands.tex` and `thesis/tex/thesis_preamble.tex`. **Reuse them —
do not invent new direction-handling macros or packages.**

- `\lr{...}` — wrap Latin/English terms embedded in Persian text so they render
  left-to-right inside the RTL paragraph flow. Used throughout, e.g.
  `\basemodel` is defined as `مدل پایه \lr{Depth Anything V2}`.
- `\rl{...}` — force right-to-left on a span (used e.g. in the draft watermark:
  `\rl{پیش‌نویس}`), and `\calibrationmatrix` wraps its Persian phrase in
  `\persianfont\rl{...}` for safety inside mixed contexts.
- `\fadecimal{...}` — wraps a decimal number so it displays LTR without
  switching to the Latin font (Persian digit glyphs aren't in the Latin font).
  Use this for every inline decimal number in Persian prose (e.g. a metric
  value like ۰٫۰۷۶), not a manually wrapped `\beginL...\endL`.
- `\enfootnote{...}` vs `\fafootnote{...}` — pick the one matching the
  footnote's language; `xepersian` handles the footnote's own RTL/LTR but the
  footnote *number* glyph is forced to Persian digits via the existing
  `\thefootnote`/`\thempfootnote` redefinition. Don't re-redefine these.
- `displaymathdigits=persian` is already set as a package option — display-math
  digits render as Persian glyphs automatically; don't manually wrap math
  numerals.
- Package load order: `xepersian` **must stay the last package loaded** in
  `commands.tex` (already documented there in a comment: "باید آخرین بسته
  باشد"). Never insert a new `\usepackage` after it, and never reorder existing
  packages — xepersian's RTL patches depend on seeing the document's other
  packages first.
- Font switching: `\if@irfonts` branches between the IRLotusICEE/Liberation
  Serif pairing and the XB Niloofar/Junicode pairing. Don't hardcode a font
  name in new content — if a new span needs a specific font, use
  `\defpersianfont`/`\deflatinfont` following the existing pattern, inside the
  same `\if@irfonts...\fi` branch if the choice should follow that toggle.
- Named model macros (`\proposedmodel`, `\controlmodel`, `\basemodel`,
  `\publishedbasemodel`, `\tableproposedmodel`, etc.) exist specifically so
  model names stay consistent and direction-safe across chapters and tables.
  Use them instead of typing the Persian/English model name inline — this is
  also a `scientific-writing` terminology-consistency requirement.
- Algorithm environments already have Persian/English conditional translations
  via `\if@RTL{...}\else ...\fi` (see the `algorithmic*` renewcommands) — don't
  add a second translation layer on top.
- Table/figure captions: `\captionsetup{font=footnotesize,labelfont=bf}` is
  already set; captions and TikZ in-figure labels follow the same `\lr{}`/
  direction rules as body text — a chart's legend text in English still needs
  `\lr{}` if it's embedded inside a Persian caption sentence.

## Procedure

1. Determine register (almost always academic Persian for `thesis/` content).
2. Draft or revise content, applying the orthography rules as you write — don't
   write first and clean up digits/ZWNJ after; it's easy to miss spans.
3. For every mixed Persian/Latin span (model names, package names, metric
   names, file paths), wrap the Latin part in `\lr{}` and confirm the sentence
   doesn't start with that Latin span.
4. For every inline decimal number, use `\fadecimal{}`.
5. Run the native-AI-tell check (banned phrase list above) before calling the
   text done.
6. If this touches `thesis/`, build via `thesis/build.sh` and visually inspect
   the rendered page for direction issues — a missing `\lr{}` often doesn't
   error, it just renders wrong, so grep alone won't catch it.

## Reporting

State the register chosen, confirm orthography rules were applied, and list any
macro (`\lr`, `\rl`, `\fadecimal`, named model macros) used or that should have
been used but wasn't available for a new term — flag if a new macro is needed
rather than hand-rolling a one-off direction fix.
