# Thesis defense deck (v2)

Offline HTML/JS slide deck for the master's thesis defense. 38 main slides + 5 backup slides, Persian, right-to-left, white and dark navy.

## Presenting

Open `index.html` in Chrome or Edge (double-click; no internet or server needed).

| Key | Action |
|---|---|
| ← / ↓ / Space / PageDown / Enter | next step or slide (the deck reads right to left, so the left arrow moves forward) |
| → / ↑ / PageUp / Backspace | previous step or slide |
| F | full screen |
| P | open the presenter window (notes, next slide, timer) |
| O | slide list; click a title to jump |
| click a part name in the top bar | jump to the first slide of that part (audience window) |
| B | jump to the first backup slide |
| Home / End | first slide / last main slide |
| H | show the key hints |
| K / J / L / R / M | video: play or pause / back 5 s / forward 5 s / from the start / sound on or off |

Presenter window: press **P** in the audience window (allow pop-ups for the file if the browser asks), drag the new window to the laptop screen and keep the audience window full screen on the projector. Either window can drive the deck; they stay in sync. The top bar of the presenter window shows the slide counter together with the current part number and name. The timer starts on the first slide change.

The video on slide 5 does not start by itself: the first press of the next key on that slide plays it, without sound, the second reveals the «small and near / large and far» pair, and the third moves on. The presenter window shows a video panel on that slide (seek bar, time, from the start, back and forward 5 s, play or pause, sound) that controls the video in the audience window; the same actions are on the K, J, L, R and M keys in either window. Letter keys are read by key position, so they also work while the Persian keyboard layout is active.

The outline tiles on slide 2, and the four application tiles and four phrases on slide 3, enter by themselves, one after another, when the slide is reached (about two seconds on slide 2, about three on slide 3); stepping back to the slide shows them all at once.

A link such as `index.html#25` opens directly on slide 25; adding `?all=1` shows a slide with all its steps revealed.

## Editing

- All Persian text (slide phrases and presenter notes) is in `content/fa.json`, keyed by slide (`s25.title`, `s25.notes`, ...).
- Slide markup is in `src/slides/*.html`; the page shell is `src/shell.html`.
- Styles are in `assets/css/` (`deck.css` engine and shared components, `slides.css` per-slide layout); the engine is `assets/js/deck.js`.
- A `<video data-play="N">` is started by step N of its slide and is paused and rewound when the slide is entered.
- A `<video data-ambient>` plays by itself, without sound and in a loop, whenever its slide is shown in the audience window, and pauses when the slide is left. The video keys and the presenter's video panel ignore it; the presenter window shows its poster image. The title slide uses this for `assets/video/depth-demo-title.mp4`.
- An element with class `auto` (plus `rise` for the upward slide) enters without a key press when its slide opens, in document order; `--auto-gap` (default 230ms) and `--auto-start` (default 320ms) on the slide set the pacing. Class `frag` with `data-f="N"` stays the key-press step.
- In the presenter notes, the English equivalent of a technical term follows its first mention on each slide in parentheses, for example `ابهام مقیاس (scale ambiguity)`; the presenter window shows these in a lighter colour and keeps each on one line. Terms follow the thesis glossary (`thesis/tex/words.tex`).
- After any change run `python3 build.py`, which regenerates `index.html` and reports missing or unused text keys.

## Sources

- Numbers, tables and chart data: `thesis/tex/chapter3.tex`, `chapter4.tex`, `chapter5.tex`.
- Qualitative and error-map images: cropped from `thesis/figures/qualitative_md_v2/*.pdf`.
- Dataset samples: `thesis/figures/samples/`. Teaser: Depth Anything V2 repository asset.
- Fonts: Vazirmatn (SIL OFL, `assets/fonts/`). Maths: KaTeX (MIT, `assets/vendor/katex/`).
