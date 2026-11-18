# Thesis defense deck (v2)

Offline HTML/JS slide deck for the master's thesis defense. 37 main slides + 4 backup slides, Persian, right-to-left, white and dark navy.

## Presenting

Open `index.html` in Chrome or Edge (double-click; no internet or server needed).

| Key | Action |
|---|---|
| → / ↓ / Space / PageDown / Enter | next step or slide (works with a clicker) |
| ← / ↑ / PageUp / Backspace | previous step or slide |
| F | full screen |
| P | open the presenter window (notes, next slide, timer) |
| O | slide list; click a title to jump |
| B | jump to the first backup slide |
| Home / End | first slide / last main slide |
| H | show the key hints |

Presenter window: press **P** in the audience window (allow pop-ups for the file if the browser asks), drag the new window to the laptop screen and keep the audience window full screen on the projector. Either window can drive the deck; they stay in sync. The timer starts on the first slide change.

The video on slide 4 starts automatically when the slide is reached and has its own controls.

A link such as `index.html#25` opens directly on slide 25; adding `?all=1` shows a slide with all its steps revealed.

## Editing

- All Persian text (slide phrases and presenter notes) is in `content/fa.json`, keyed by slide (`s25.title`, `s25.notes`, ...).
- Slide markup is in `src/slides/*.html`; the page shell is `src/shell.html`.
- Styles are in `assets/css/` (`deck.css` engine and shared components, `slides.css` per-slide layout); the engine is `assets/js/deck.js`.
- After any change run `python3 build.py`, which regenerates `index.html` and reports missing or unused text keys.

## Sources

- Numbers, tables and chart data: `thesis/tex/chapter3.tex`, `chapter4.tex`, `chapter5.tex`.
- Qualitative and error-map images: cropped from `thesis/figures/qualitative_md_v2/*.pdf`.
- Dataset samples: `thesis/figures/samples/`. Teaser: Depth Anything V2 repository asset.
- Fonts: Vazirmatn (SIL OFL, `assets/fonts/`). Maths: KaTeX (MIT, `assets/vendor/katex/`).
