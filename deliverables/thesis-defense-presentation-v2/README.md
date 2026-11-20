# Thesis defense deck (v2)

**This is the main defense deck.** All slide and presenter-note updates go here. The other decks under `deliverables/` are left untouched.

Offline HTML/JS slide deck for the master's thesis defense. 39 main slides + 8 backup slides, Persian, right-to-left, white and dark navy.

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

The **خلاصه** switch above the presenter notes shows a short cue-card version of each slide's notes (numbered steps, bullets, key numbers, and a highlighted caveat line). It changes only when clicked: slide changes never reset it, and the choice is remembered when the presenter window is reopened. With the switch off, the full notes are shown in short paragraphs. In both views numbers are highlighted.

The video on slide 5 does not start by itself: the first press of the next key on that slide plays it, without sound, the second reveals the «small and near / large and far» pair, and the third moves on. The presenter window shows a video panel on that slide (seek bar, time, from the start, back and forward 5 s, play or pause, sound) that controls the video in the audience window; the same actions are on the K, J, L, R and M keys in either window. Letter keys are read by key position, so they also work while the Persian keyboard layout is active.

The outline tiles on slide 2, and the four application tiles and four phrases on slide 3, enter by themselves, one after another, when the slide is reached (about two seconds on slide 2, about three on slide 3); stepping back to the slide shows them all at once.

A link such as `index.html#25` opens directly on slide 25; adding `?all=1` shows a slide with all its steps revealed.

## Presenting from the phone

```bash
python3 serve.py              # or: python3 serve.py --port 9000
```

Only the Python standard library is used. The server prints three things:

- the audience-window address, `http://localhost:8000/`: open it in Chrome or Edge on the laptop, press F and put it on the projector;
- a QR code, both in the terminal and at `http://localhost:8000/qr`: scan it with the phone camera;
- the phone link itself, `http://<laptop-ip>:8000/?remote=1&k=<token>`.

The phone opens a presenter view laid out for the right hand. From top to bottom: the bar (slide counter, part, timer, full screen), the notes (with the **خلاصه** switch and A−/A+ text size, remembered on the phone), the current slide, and at the bottom right the picture of what comes next. That picture is the **بعدی** button: tap anywhere on it to move on. To its left are **قبلی** and **فهرست** (the slide list; tap a title to jump, or «بستن» to close it). Touching the current-slide picture, and dragging on it, shows a red pointer dot at the same place on the audience screen; it disappears when the finger is lifted. On the video slide the video panel appears between the two slides. With the phone on its side the notes are on the left and the slides and buttons on the right. The page keeps the screen awake after the first tap. The keyboard on the laptop and the laptop presenter window (P) still work, and every view stays in sync.

A dot at the top shows the link. If the audience window is closed, or the server can no longer be reached, a yellow or red banner appears and the buttons dim. Reloading the phone page always picks up the current slide.

- The phone and the laptop must be on the same network. University Wi-Fi often blocks devices from talking to each other: if the phone cannot open the page, turn on the phone's hotspot, connect the laptop to it, and restart `serve.py` so it prints the new address.
- On Windows, allow Python through the firewall when asked (private network).
- The token in the link changes each time the server starts, so scan again after a restart. `--token abc` keeps it fixed. Anyone on the network can view the slides, but only someone with the link can control them.

## Editing

- All Persian text (slide phrases and presenter notes) is in `content/fa.json`, keyed by slide (`s25.title`, `s25.notes`, ...). The quick-glance summary of a slide's notes is `s25.sum`: one item per line, `1. ` numbered, `- ` bullet, `! ` caveat.
- Slide markup is in `src/slides/*.html`; the page shell is `src/shell.html`.
- Styles are in `assets/css/` (`deck.css` engine and shared components, `slides.css` per-slide layout); the engine is `assets/js/deck.js`.
- A `<video data-play="N">` is started by step N of its slide and is paused and rewound when the slide is entered.
- A `<video data-ambient>` plays by itself, without sound and in a loop, whenever its slide is shown in the audience window, and pauses when the slide is left. The video keys and the presenter's video panel ignore it; the presenter window shows its poster image. The title slide uses this for `assets/video/depth-demo-title.mp4`.
- An element with class `auto` (plus `rise` for the upward slide) enters without a key press when its slide opens, in document order; `--auto-gap` (default 230ms) and `--auto-start` (default 320ms) on the slide set the pacing. Class `frag` with `data-f="N"` stays the key-press step.
- In the presenter notes, the English equivalent of a technical term follows its first mention on each slide in parentheses, for example `ابهام مقیاس (scale ambiguity)`; the presenter window shows these in a lighter colour and keeps each on one line. Terms follow the thesis glossary (`thesis/tex/words.tex`).
- Abbreviations in the notes are spelled out in full at their first mention on each slide, for example `DPT (Dense Prediction Transformer)`; inside a parenthesis the form is `(Full Name, ABBR 2017)`. The English spans in the presenter window wrap as a block, so long names stay readable.
- Formulas in the notes are written as `$...$` (KaTeX syntax) inside the text in `content/fa.json`. The presenter window typesets them left-to-right; short ones stay in the line, long ones (over 26 characters) get a centred line of their own. Metric definitions (AbsRel, RMSE, δ₁) are in the notes of the «سه معیار، چهار مجموعه آزمون» slide and backup 3; the scale-and-shift alignment is in the notes of «ارزیابی نسبی و مطلق؛ سهم دوربین». Backup slides 6 and 7 are formula sheets for questions: every formula of the thesis (metrics incl. SqRel/log10/MAE, per-image alignment closed form, SSI pixel and gradient terms, SiLog and its variance form; camera model, descriptor, gate token, zoom-crop, focal jitter, lr schedule, attention), with notes saying where each one is used. In b6 the SiLog log-difference is written $e_i$ (thesis: $d_i$) to avoid clashing with inverse depth $d_i$.
- Slides 8 and 9 are the era timeline (keys `s8.*` and `s8b.*`): four era cards each, one click per card. The pictures `assets/img/tl-1..7.jpg` are crops of figures from the cited papers (Photo Pop-up, Make3D, Eigen et al., Monodepth, MiDaS, Depth Anything, Metric3D); `tl-8.jpg` is our own NYUv2 result. The story and data of each era are in the notes, not on the slide.
- The last backup slide (keys `b8.*`) is «Depth Anything 3: معماری و ایده‌های اصلی»: a flow diagram of DA3 (single plain transformer, optional camera token, Dual-DPT head giving depth and ray maps), four focus phrases and a size/speed table taken from the DA3 paper (arXiv 2511.10647). The full explanation (architecture, depth-ray target, teacher-student training, results, ablations, differences from this work) is in its notes. It is not shown in the talk.
- After any change run `python3 build.py`, which regenerates `index.html` and reports missing or unused text keys.

## Sources

- Numbers, tables and chart data: `thesis/tex/chapter3.tex`, `chapter4.tex`, `chapter5.tex`.
- Qualitative and error-map images: cropped from `thesis/figures/qualitative_md_v2/*.pdf`.
- Dataset samples: `thesis/figures/samples/`. Teaser: Depth Anything V2 repository asset.
- Fonts: Vazirmatn (SIL OFL, `assets/fonts/`). Maths: KaTeX (MIT, `assets/vendor/katex/`).
