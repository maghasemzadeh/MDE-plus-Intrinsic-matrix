# Thesis defense deck (v2)

**This is the main defense deck.** All slide and presenter-note updates go here. The other decks under `deliverables/` are left untouched.

Offline HTML/JS slide deck for the master's thesis defense. 40 main slides + 13 backup slides, Persian, right-to-left, white and dark navy.

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

Pointer: in the presenter window, hold the left mouse button on the current-slide picture and move; a red dot follows at the same place on the audience screen and disappears on release. The phone view does the same with a finger. The dot leaves a red line behind it that thins and fades over about two seconds (`TRAIL_MS` in `assets/js/deck.js`), so a shape drawn by hand, such as a circle around a number, stays visible for a moment.

Presenter window: press **P** in the audience window (allow pop-ups for the file if the browser asks), drag the new window to the laptop screen and keep the audience window full screen on the projector. Either window can drive the deck; they stay in sync. The top bar of the presenter window shows the slide counter together with the current part number and name. The timer starts on the first slide change.

The **خلاصه** switch above the presenter notes shows a short cue-card version of each slide's notes (numbered steps, bullets, key numbers, and a highlighted caveat line). It changes only when clicked: slide changes never reset it, and the choice is remembered when the presenter window is reopened. With the switch off, the full notes are shown in short paragraphs. In both views numbers are highlighted.

The video on slide 5 is trimmed to its first 6 seconds and is an ambient video (`data-ambient`): it starts by itself, without sound, when the slide opens and loops; the first press of the next key reveals the «small and near / large and far» pair and the second moves on. Because it is ambient, the presenter's video panel and the K/J/L/R/M keys do not control it.

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

After the deck files change, reload the laptop page as well as the phone page. If the laptop page is still an older version, the phone shows «صفحهٔ لپ‌تاپ قدیمی است؛ آن را دوباره بارگذاری کنید».

**S Pen as a clicker (optional, not yet confirmed on the phone).** The pen button at the top of the phone view turns pen mode on; the choice is remembered, and after a reload it starts on the first tap. A web page cannot read the S Pen button directly, so the page listens for what an Air action can turn a press into: media keys (single press = play/pause, double press = next track; a silent sound loop runs so the phone sends them to this page), the browser's Back and Forward, and ordinary keys such as Page Down. The first of each pair moves to the next step, the second goes back, and a green label names what arrived (`قلم: بعدی · back`, `· pause`, ...). On the phone, check Settings → Advanced features → S Pen → Air actions, and under the browser's app actions set single press and double press so that the label shows next for one press and previous for two. While pen mode is on, the phone's own Back gesture also means next instead of leaving the page.

A dot at the top shows the link. If the audience window is closed, or the server can no longer be reached, a yellow or red banner appears and the buttons dim. Reloading the phone page always picks up the current slide.

- The phone and the laptop must be on the same network. University Wi-Fi often blocks devices from talking to each other: if the phone cannot open the page, turn on the phone's hotspot, connect the laptop to it, and restart `serve.py` so it prints the new address.
- On Windows, allow Python through the firewall when asked (private network).
- The token in the link changes each time the server starts, so scan again after a restart. `--token abc` keeps it fixed. Anyone on the network can view the slides, but only someone with the link can control them.

## Editing

- Notes can be edited by hand in the laptop presenter window when the deck is opened through `serve.py` (`http://localhost:8000/`, then P). **ویرایش** opens the stored text of the view that is showing: the full notes, or with the **خلاصه** switch on, the summary lines. **ذخیره** (or Ctrl+Enter) writes it to `content/fa.json` and rebuilds `index.html`; **انصراف** (or Esc) discards it. The text is stored and displayed exactly as before: formulas stay `$...$`, and summary lines keep their `1. ` / `- ` / `! ` markers. An open phone view updates at once. The phone cannot edit, and a deck opened as a file (no server) shows the button disabled.

- All Persian text (slide phrases and presenter notes) is in `content/fa.json`, keyed by slide (`s25.title`, `s25.notes`, ...). The quick-glance summary of a slide's notes is `s25.sum`: one item per line, `1. ` numbered, `- ` bullet, `! ` caveat.
- Slide markup is in `src/slides/*.html`; the page shell is `src/shell.html`.
- Styles are in `assets/css/` (`deck.css` engine and shared components, `slides.css` per-slide layout); the engine is `assets/js/deck.js`.
- A `<video data-play="N">` is started by step N of its slide and is paused and rewound when the slide is entered.
- A `<video data-ambient>` plays by itself, without sound and in a loop, whenever its slide is shown in the audience window, and pauses when the slide is left. The video keys and the presenter's video panel ignore it; the presenter window shows its poster image. The title slide uses this for `assets/video/depth-demo-title.mp4`.
- An element with class `auto` (plus `rise` for the upward slide) enters without a key press when its slide opens, in document order; `--auto-gap` (default 450ms, so each item finishes its 420ms entrance before the next starts, like a key-press step) and `--auto-start` (default 320ms) on the slide set the pacing. Class `frag` with `data-f="N"` stays the key-press step.
- A slide with `data-autostep` (after its `class` attribute) runs its `frag` steps by itself when reached moving forward: the first one after about 0.6 s (later if the slide has `auto` items), then one every 900 ms (`data-autostep="1200"` sets another gap). Only the audience window drives it; the presenter and phone views follow. A key press still steps at once and the rest carries on from there; after stepping back nothing advances by itself. Slides 20–38 have it; remove it from a slide to make that slide click-driven again.
- In the presenter notes, the English equivalent of a technical term follows its first mention on each slide in parentheses, for example `ابهام مقیاس (scale ambiguity)`; the presenter window shows these in a lighter colour and keeps each on one line. Terms follow the thesis glossary (`thesis/tex/words.tex`).
- Abbreviations in the notes are spelled out in full at their first mention on each slide, for example `DPT (Dense Prediction Transformer)`; inside a parenthesis the form is `(Full Name, ABBR 2017)`. The English spans in the presenter window wrap as a block, so long names stay readable.
- Formulas in the notes are written as `$...$` (KaTeX syntax) inside the text in `content/fa.json`. The presenter window typesets them left-to-right; short ones stay in the line, long ones (over 26 characters) get a centred line of their own. Metric definitions (AbsRel, RMSE, δ₁) are in the notes of the «سه معیار، چهار مجموعه آزمون» slide and backup 3; the scale-and-shift alignment is in the notes of «ارزیابی نسبی و مطلق؛ سهم دوربین». Backup slides 5 and 6 (keys `b6.*`, `b7.*`) are formula sheets for questions: every formula of the thesis (metrics incl. SqRel/log10/MAE, per-image alignment closed form, SSI pixel and gradient terms, SiLog and its variance form; camera model, descriptor, gate token, zoom-crop, focal jitter, lr schedule, attention), with notes saying where each one is used. In b6 the SiLog log-difference is written $e_i$ (thesis: $d_i$) to avoid clashing with inverse depth $d_i$.
- Slides 8 and 9 are the era timeline (keys `s8.*` and `s8b.*`): four era cards each, one click per card. The pictures `assets/img/tl-1..7.jpg` are crops of figures from the cited papers (Photo Pop-up, Make3D, Eigen et al., Monodepth, MiDaS, Depth Anything, Metric3D); `tl-8.jpg` is our own NYUv2 result. The story and data of each era are in the notes, not on the slide.
- Slide 35 (keys `s34.*`) shows the encoder-size and model-type result as two same-scale horizontal bar charts, ViT-S and ViT-L (`data-chart="hbars"` in `deck.js`): per test set, how much lower the proposed model's AbsRel is than its control's (%), navy for absolute models and gray for relative ones, with a ±3% band and a star for Wilcoxon p < 0.01. The relative bars and their message come in on the one key-press step. The full table with AbsRel values is backup 2.
- Slide 39 (keys `sref.*`), just before the thank-you slide, is «منابع»: only the ten most important works (Eigen et al., MiDaS, DPT, Depth Anything V2, DINOv2; CAM-Convs, Metric3D, UniDepth, NYUv2, KITTI) in two columns (`sref.k1`, `sref.k2`). The full list of works named on the main slides is in the last four backup slides, 10–13 (keys `b10.*`–`b13.*`), one group per slide: the methods timeline, the base model and encoder, camera-aware and metric-depth methods, and datasets, AdamW and the Wilcoxon test (group names `sref.g1`–`sref.g4`). It is not the thesis's full bibliography. Entries follow `thesis/tex/MyReferences.bib`, except Automatic Photo Pop-up (Hoiem et al., SIGGRAPH 2005), which the timeline names but the thesis does not cite.
- Backup 1 (keys `b1.*`) has two tables: the left one holds the test splits of the training datasets (NYUv2, Cityscapes, KITTI), the right one the datasets not seen in training (SUN RGB-D Kinect v1, Kinect v2, RealSense, Xtion, and DIODE).
- Backup 7 (keys `b5.*`) is «مقایسه با Depth Anything 3», placed directly before backup 8 so the two DA3 slides sit together; its notes point to the next slide for the architecture.
- Backup 8 (keys `b8.*`) is «Depth Anything 3: معماری و ایده‌های اصلی»: a flow diagram of DA3 (single plain transformer, optional camera token, Dual-DPT head giving depth and ray maps), four focus phrases and a size/speed table taken from the DA3 paper (arXiv 2511.10647). The full explanation (architecture, depth-ray target, teacher-student training, results, ablations, differences from this work) is in its notes. It is not shown in the talk.
- Backup 9 (keys `b9.*`) is «منحنی آموزش مدل نهایی»: per-epoch mean validation AbsRel (epochs 1–12) and mean training loss (epochs 2–12; epoch 1 in the footnote) of the final ViT-L multi-domain pair, proposed vs control. The two charts are static SVG generated from `~/curves/curves.json` (parsed from lingotube `logs/md1-P.log` / `md1-C.log`); the notes say plainly that the late flattening coincides with the learning rate reaching about zero. Slide 19's notes point to it.
- After any change run `python3 build.py`, which regenerates `index.html` and reports missing or unused text keys.
- Live editing: open `http://localhost:8000/?live=1` (through `serve.py`). The page reloads by itself within about a second when `index.html`, `deck.css`, `slides.css` or `deck.js` changes, and stays on the same slide. Without `?live=1` nothing changes.

## Sources

- Numbers, tables and chart data: `thesis/tex/chapter3.tex`, `chapter4.tex`, `chapter5.tex`.
- Qualitative and error-map images: cropped from `thesis/figures/qualitative_md_v2/*.pdf`.
- Dataset samples: `thesis/figures/samples/`. Teaser: Depth Anything V2 repository asset.
- Fonts: Vazirmatn (SIL OFL, `assets/fonts/`). Maths: KaTeX (MIT, `assets/vendor/katex/`).
