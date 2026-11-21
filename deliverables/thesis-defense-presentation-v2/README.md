# Thesis defense deck (v2)

**This is the main defense deck.** All slide and presenter-note updates go here. The other decks under `deliverables/` are left untouched.

Offline HTML/JS slide deck for the master's thesis defense. 35 main slides + 15 backup slides, Persian, right-to-left, white and dark navy.

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
- VPN addresses are never put in the QR code or the printed links: `240.x.x.x` and `10.119.20.x` are always skipped (the phone is not on the VPN, so such a link cannot open), and every other adapter address is offered instead, Wi-Fi and hotspot first. `--skip-ip 10.8.0.0/24` skips one more address or network, and `--ip 192.168.1.50` forces a single address. If the phone still cannot open the link with the VPN connected, the VPN is blocking local-network traffic: disconnect it while presenting.
- On Windows, allow Python through the firewall when asked (private network).
- The token in the link changes each time the server starts, so scan again after a restart. `--token abc` keeps it fixed. Anyone on the network can view the slides, but only someone with the link can control them.

## Editing

- Notes can be edited by hand in the laptop presenter window when the deck is opened through `serve.py` (`http://localhost:8000/`, then P). **ویرایش** opens the stored text of the view that is showing: the full notes, or with the **خلاصه** switch on, the summary lines. **ذخیره** (or Ctrl+Enter) writes it to `content/fa.json` and rebuilds `index.html`; **انصراف** (or Esc) discards it. The text is stored and displayed exactly as before: formulas stay `$...$`, and summary lines keep their `1. ` / `- ` / `! ` markers. An open phone view updates at once. The phone cannot edit, and a deck opened as a file (no server) shows the button disabled.

- All Persian text (slide phrases and presenter notes) is in `content/fa.json`, keyed by slide (`s26.title`, `s26.notes`, ...). The quick-glance summary of a slide's notes is `s26.sum`: one item per line, `1. ` numbered, `- ` bullet, `! ` caveat.
- Slide markup is in `src/slides/*.html`; the page shell is `src/shell.html`.
- Styles are in `assets/css/` (`deck.css` engine and shared components, `slides.css` per-slide layout); the engine is `assets/js/deck.js`.
- A `<video data-play="N">` is started by step N of its slide and is paused and rewound when the slide is entered.
- A `<video data-ambient>` plays by itself, without sound and in a loop, whenever its slide is shown in the audience window, and pauses when the slide is left. The video keys and the presenter's video panel ignore it; the presenter window shows its poster image. The title slide uses this for `assets/video/depth-demo-title.mp4`.
- An element with class `auto` (plus `rise` for the upward slide) enters without a key press when its slide opens, in document order; `--auto-gap` (default 450ms, so each item finishes its 420ms entrance before the next starts, like a key-press step) and `--auto-start` (default 320ms) on the slide set the pacing. Class `frag` with `data-f="N"` stays the key-press step.
- A slide with `data-autostep` (after its `class` attribute) runs its `frag` steps by itself when reached moving forward: the first one after about 0.6 s (later if the slide has `auto` items), then one every 900 ms (`data-autostep="1200"` sets another gap). Only the audience window drives it; the presenter and phone views follow. A key press still steps at once and the rest carries on from there; after stepping back nothing advances by itself. The results slides (20–30) and slides 32–33 have it; remove it from a slide to make that slide click-driven again.
- In the presenter notes, the English equivalent of a technical term follows its first mention on each slide in parentheses, for example `ابهام مقیاس (scale ambiguity)`; the presenter window shows these in a lighter colour and keeps each on one line. Terms follow the thesis glossary (`thesis/tex/words.tex`).
- Abbreviations in the notes are spelled out in full at their first mention on each slide, for example `DPT (Dense Prediction Transformer)`; inside a parenthesis the form is `(Full Name, ABBR 2017)`. The English spans in the presenter window wrap as a block, so long names stay readable.
- Formulas in the notes are written as `$...$` (KaTeX syntax) inside the text in `content/fa.json`. The presenter window typesets them left-to-right; short ones stay in the line, long ones (over 26 characters) get a centred line of their own. Metric definitions (AbsRel, RMSE, δ₁) are in the notes of the «سه معیار، چهار مجموعه آزمون» slide and backup 4; the scale-and-shift alignment is in the notes of «ارزیابی نسبی و مطلق؛ سهم دوربین». Backup slides 6 and 7 (keys `b6.*`, `b7.*`) are formula sheets for questions: every formula of the thesis (metrics incl. SqRel/log10/MAE, per-image alignment closed form, SSI pixel and gradient terms, SiLog and its variance form; camera model, descriptor, gate token, zoom-crop, focal jitter, lr schedule, attention), with notes saying where each one is used. In b6 the SiLog log-difference is written $e_i$ (thesis: $d_i$) to avoid clashing with inverse depth $d_i$.
- Parts and slide order (2026-11-20 restructure): 1 problem (3–7), 2 background (8–10), 3 method (11–14: idea, descriptor, architecture, gate), 4 «مجموعه‌داده» (15 datasets, 16 zoom-crop, 17 focal jitter), 5 «آموزش» (18 paired design, 19 training recipe), 6 «نتایج» (20–30), 7 «جمع‌بندی» (31–35). On 2026-11-21 the prior-methods table «دوربین در روش‌های پیشین؛ شکاف پژوهشی» (keys `s10.*`, class `s-prior`) left part 2 and became backup 1, shown whole with no click steps; every later main slide moved up by one and every older backup slide down by one. The talk presents the final multi-dataset model directly: the four-phase slide, the pipeline-verification slide, the final-model diagram, the relative-to-metres table and the per-image wins slide were removed, with their text keys (`s19`, `s20`, `s25`, `s29`, `s33`). The `sN.` keys are not slide numbers: a key keeps its name when slides move (`s15` is slide 18, `s26` is slide 26, `s34` is slide 22).
- Slides 16 and 17 (keys `s17.*`, `s17b.*`) are the two camera-aware augmentations, one per slide. The zoom-crop picture shows a zoom of 2, the upper end of the final model's range (63° becomes about 34°); the 1–2.2 range of the single-camera NYUv2 study is only in the notes. The focal-jitter slide shows the same picture twice, with only the focal length and the marked depths multiplied by α.
- Slide 20 (keys `sq1.*`), the first results slide, answers research question 1: the released Depth Anything V2 metric model (ViT-L, Hypersim checkpoint, no alignment) on five indoor test sets ordered by horizontal field of view (SUN RGB-D RealSense, Xtion, Kinect v1; NYUv2; SUN RGB-D Kinect v2). Bars are AbsRel; the pill and dot are the quartiles and median of the per-image scale ratio median(pred)/median(gt). The chart is static SVG; its numbers come from `results/multidomain/released_hypersim_vitl` and `released_hypersim_vitl_sunrgbd` (the AbsRel values are those of backup 2; the scale ratios and the two-sample t-test, RealSense vs Kinect v2, were computed from the per-image records and are not printed in the thesis).
- Slides 21 and 23–25 (relative vs absolute evaluation, focal-jitter leap, focal-length sweep, calibration knockout) show the single-camera NYUv2 controlled study and say so in their subtitle (`g.ctrl_study`); slide 22, between them, is the multi-dataset encoder-size and model-type chart (keys `s34.*`), placed right after the relative-vs-absolute evaluation slide; slide 26 onward is the final model (`s26.sub`).
- Each research question is answered on one results slide by a navy chip: question 1 on slide 20 (`sq1.answer`), question 2 on slide 22 (`s34.answer`), question 3 on slide 27, «سه معیار، چهار مجموعه آزمون» (`s27.answer`, the second step of that slide).
- Slides 8 and 9 are the era timeline (keys `s8.*` and `s8b.*`): four era cards each, one click per card. The pictures `assets/img/tl-1..7.jpg` are crops of figures from the cited papers (Photo Pop-up, Make3D, Eigen et al., Monodepth, MiDaS, Depth Anything, Metric3D); `tl-8.jpg` is our own NYUv2 result. The story and data of each era are in the notes, not on the slide.
- Slide 22 (keys `s34.*`) shows the encoder-size and model-type result as two same-scale horizontal bar charts, ViT-S and ViT-L (`data-chart="hbars"` in `deck.js`): per test set, how much lower the proposed model's AbsRel is than its control's (%), navy for absolute models and gray for relative ones, with a ±3% band and a star for Wilcoxon p < 0.01. The relative bars and their message come in on the one key-press step. The full table with AbsRel values is backup 3.
- Slide 34 (keys `sref.*`), just before the thank-you slide, is «منابع»: only the ten most important works (Eigen et al., MiDaS, DPT, Depth Anything V2, DINOv2; CAM-Convs, Metric3D, UniDepth, NYUv2, KITTI) in two columns (`sref.k1`, `sref.k2`). The full list of works named on the main slides is in backup slides 11–14 (keys `b10.*`–`b13.*`), one group per slide: the methods timeline, the base model and encoder, camera-aware and metric-depth methods, and datasets, AdamW and the Wilcoxon test (group names `sref.g1`–`sref.g4`). It is not the thesis's full bibliography. Entries follow `thesis/tex/MyReferences.bib`, except Automatic Photo Pop-up (Hoiem et al., SIGGRAPH 2005), which the timeline names but the thesis does not cite.
- Backup 2 (keys `b1.*`) has two tables: the left one holds the test splits of the training datasets (NYUv2, Cityscapes, KITTI), the right one the datasets not seen in training (SUN RGB-D Kinect v1, Kinect v2, RealSense, Xtion, and DIODE).
- Backup 8 (keys `b5.*`) is «مقایسه با Depth Anything 3», placed directly before backup 9 so the two DA3 slides sit together; its notes point to the next slide for the architecture.
- Backup 9 (keys `b8.*`) is «Depth Anything 3: معماری و ایده‌های اصلی»: a flow diagram of DA3 (single plain transformer, optional camera token, Dual-DPT head giving depth and ray maps), four focus phrases and a size/speed table taken from the DA3 paper (arXiv 2511.10647). The full explanation (architecture, depth-ray target, teacher-student training, results, ablations, differences from this work) is in its notes. It is not shown in the talk.
- Backup 10 (keys `b9.*`) is «منحنی آموزش مدل نهایی»: per-epoch mean validation AbsRel (epochs 1–12) and mean training loss (epochs 2–12; epoch 1 in the footnote) of the final ViT-L multi-domain pair, proposed vs control. The two charts are static SVG generated from `~/curves/curves.json` (parsed from lingotube `logs/md1-P.log` / `md1-C.log`); the notes say plainly that the late flattening coincides with the learning rate reaching about zero. Slide 19's notes point to it.
- Backup 15 (keys `b14.*`, class `s-b14`), the last slide of the deck, is «درون انکودر دوربین»: the modules of the camera encoder in execution order as in `cam_enc.py` (9-D descriptor, input MLP 9→512→1024, LayerNorm, 4 transformer blocks, LayerNorm, gate, sum with CLS) and, in the lower panel, the inside of one block (attention branch and MLP branch, each with LayerScale and a residual connection). Dimensions are for ViT-L. Parameter counts on the chips and in the notes were computed from the layer shapes (ViT-L 50.9M total, 12.6M per block; ViT-S 7.2M) and match the model totals in the encoder-size slide's notes. The notes also hold the single-token point (with one token the attention weight is always 1, so self-attention is a linear map) and the final model's gate value (about −0.034, from `results/multidomain/md1_P`); neither is on the slide face. The architecture slide's notes point to it.
- After any change run `python3 build.py`, which regenerates `index.html` and reports missing or unused text keys.
- Live editing: open `http://localhost:8000/?live=1` (through `serve.py`). The page reloads by itself within about a second when `index.html`, `deck.css`, `slides.css` or `deck.js` changes, and stays on the same slide. Without `?live=1` nothing changes.

## Sources

- Numbers, tables and chart data: `thesis/tex/chapter3.tex`, `chapter4.tex`, `chapter5.tex`.
- Qualitative and error-map images (slides 29 and 30): `assets/img/qual-rows.jpg` and `assets/img/error-rows.jpg`, written by `make_qual_images.py` from `results/qualitative_md_v2/candidates` (no GPU). One row per test set (NYUv2, SUN RGB-D Kinect v2, Cityscapes): the first image of each set in `thesis/figures/qualitative_md_v2/manifest.json`, i.e. images selected after evaluation. Both files have the same size and panel boxes, so the two slides show the same scenes at the same places; the error image keeps the input and reference-depth columns and replaces the three model columns with the predicted/reference ratio. The script needs matplotlib, OpenCV and a TTF copy of Vazirmatn (`--font-dir`); it prints the column and row centres used as `left:%` / `top:%` in the slide markup. The former two depth-map slides (keys `s31.*`) were merged into slide 29.
- Dataset samples: `thesis/figures/samples/`. Teaser: Depth Anything V2 repository asset.
- Fonts: Vazirmatn (SIL OFL, `assets/fonts/`). Maths: KaTeX (MIT, `assets/vendor/katex/`).
