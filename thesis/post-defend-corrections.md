# Post-defense corrections

The thesis has been submitted and cannot be changed before the defense. This file collects the
corrections to apply afterwards, in the corrected version handed in after the defense session.
Nothing listed here has been applied to `thesis/tex/`.

Line numbers refer to the submitted source (branch `refactor/final`, 2026-11-19). They will drift,
so each item also quotes the text to search for.

---

## C1 — Say "camera-induced scale ambiguity", not "scale ambiguity", wherever K is said to resolve or measure it

Added 2026-11-19.

### Why

A single image carries two different ambiguities, and the thesis uses one name, «ابهام مقیاس», for both.

1. **Size versus distance** (Figure 2-1, panel الف). A small near object and a large far object give
   the same image with the same camera. This is inherent in a single image. No single-image method
   removes it, with or without K; a model only reduces it with what it has learned about the size of
   things.
2. **Focal length versus distance** (Figure 2-1, panel ب). The same object at depth Z with focal
   length f, and at depth αZ with focal length αf, give the same image. This part comes from the
   camera, and it is the only part the calibration matrix removes.

The pinhole relation makes the split explicit: size in pixels = f × real size / distance. The image
gives the size in pixels, the network supplies the real size from learned priors, and K supplies f.
K is necessary for depth in metres; it is not sufficient alone.

Every experiment in the thesis (focal jitter, focal-length sweep, removing or distorting K, the
paired multi-dataset comparison) is about the second ambiguity. Sentences that say the proposed
model resolves or measures «ابهام مقیاس» without qualification can therefore be read as a claim
about the first one, which the thesis does not make and no single-image method could support.

### Wording to use

Metric3D (Yin et al., ICCV 2023, already cited as `Yin2023Metric3D`) names the second ambiguity
this way:

- abstract: "resolving the metric ambiguity from various camera models";
- Section 3.1, "Metric Ambiguity Analysis": "The focal length is vital for metric depth estimation."

The thesis already has a Persian phrase for it, in `chapter2.tex` line 265:
«ابهام مقیاس ناشی از دوربین». Use that phrase everywhere below, and give the Metric3D wording
"metric ambiguity from various camera models" as its English equivalent at the first mention.

Keep «ابهام مقیاس» alone only where the text means the inherent size-versus-distance ambiguity or
the unknown scale of a relative-depth output.

### Must change (the sentence claims K or the proposed model resolves or measures the ambiguity)

1. `chapter3.tex` line 487 — Figure «افزونه‌سازی لرزش کانونی», box of the proposed model.
   - Now: «… ⇒ ابهام مقیاس به‌طور کامل رفع می‌گردد»
   - Change to: «… ⇒ ابهام مقیاس ناشی از دوربین رفع می‌گردد»
   - This is the strongest overclaim in the thesis: read alone it says the scale ambiguity is
     completely resolved.

2. `chapter3.tex` line 444 — §«افزونه‌سازی لرزش کانونی».
   - Now: «… می‌تواند α را مستقیما از مقدار فاصله کانونی بخواند و ابهام را به‌طور کامل رفع کند.»
   - Change to: «… و این ابهام مقیاس ناشی از دوربین را رفع کند.»

3. `chapter5.tex` lines 196–199 — §«نوآوری‌های تحقیق».
   - Now: «صورت‌بندی هندسی ابهام مقیاس، به‌خودی‌خود تازه نیست و پیش‌تر در Metric3D … ارائه شده است. سهم نظری پژوهش حاضر، اندازه‌گیری این ابهام است»
   - Change «ابهام مقیاس» to «ابهام مقیاس ناشی از دوربین» and add the English equivalent
     "metric ambiguity from various camera models" here, since this is the sentence that cites
     Metric3D for it.

4. `chapter5.tex` lines 250–251 — contributions table, first theoretical row.
   - Now: «بازتعریف ابهام مقیاس به‌عنوان محدودیت هندسی» and «ابهام مقیاس یک ویژگی ذاتی نگاشت تصویربرداری است، نه ضعف معماری یا کمبود داده؛ بنابراین با بزرگ‌ترکردن شبکه رفع نمی‌شود»
   - Change both to «ابهام مقیاس ناشی از دوربین». The supporting evidence in that row (Figure 2-1
     and the K removal test) is evidence for the camera part only.

5. `chapter5.tex` line 44 — §«جمع‌بندی یافته‌های اصلی پژوهش».
   - Now: «ابهام مقیاس، نه یک ضعف پیاده‌سازی، بلکه یک محدودیت هندسی در مدل‌هایی است که صرفا بر اطلاعات تصویری تکیه دارند»
   - Change to «ابهام مقیاس ناشی از دوربین، …».

6. `chapter4.tex` line 1088 — §«یافته‌های اصلی و دامنه ادعای پژوهش».
   - Now: «ابهام مقیاس در آموزش چنددوربینه ریشه هندسی دارد و نه ریشه معماری.»
   - Change to «ابهام مقیاس ناشی از دوربین در آموزش چنددوربینه …».

### Should change (the ambiguity meant is the one created by changing the focal length)

7. `chapter4.tex` line 452 — §«پیکربندی لرزش کانونی».
   - Now: «برای ایجاد بیشترین ابهام در مقیاس، افزونه‌سازی لرزش کانونی … اضافه شد.»
   - Change to «برای ایجاد بیشترین ابهام مقیاس ناشی از دوربین، …».

8. `chapter4.tex` line 642 — caption of the dose-response figure.
   - Now: «افزایش سهم ماتریس کالیبراسیون دوربین با افزایش شدت ابهام مقیاس»
   - Change to «… با افزایش شدت ابهام مقیاس ناشی از دوربین». The list of figures changes with it.

9. `chapter4.tex` line 646 — paragraph under that figure.
   - Now: «با افزایش ابهام مقیاس تحمیل‌شده به داده‌های آموزشی، …»
   - Change to «با افزایش ابهام مقیاس ناشی از دوربین که به داده‌های آموزشی تحمیل شد، …».

10. `chapter4.tex` line 1055 — §«وابستگی سهم پارامترهای دوربین به پروتکل ارزیابی».
    - Now: «افزایش این برتری با افزایش شدت ابهام مقیاس»
    - Change to «… شدت ابهام مقیاس ناشی از دوربین».

11. `chapter2.tex` line 280 — DMD paragraph.
    - Now: «حتی یک اسکالر FoV می‌تواند ابهام مقیاس را کاهش دهد»
    - Change to «… ابهام مقیاس ناشی از دوربین را کاهش دهد».

12. `chapter2.tex` line 278 — Metric3D paragraph.
    - Now: «Metric3D ابهام را با نگاشت ورودی‌ها به فضای دوربین متعارف … کاهش می‌دهد»
    - Change to «Metric3D ابهام مقیاس ناشی از دوربین را …», matching the paper's own wording.

13. `chapter2.tex` lines 17 and 64 — sentence introducing Figure 2-1 and the title of panel ب.
    - Now: the sentence says the figure shows «ابهام مقیاس … و نیاز به ماتریس مشخصه (کالیبراسیون) دوربین»; panel ب is titled «(ب) هم‌ارزی دقیق کانون و مقیاس».
    - Change: say that panel الف shows the scale ambiguity inherent in a single image and panel ب
      shows the camera-induced scale ambiguity, which the calibration matrix removes. Add the term
      to the panel ب title. The figure caption (line 83) can stay.

14. `chapter1.tex` lines 65–66 — §«تعریف موضوع تحقیق».
    - Now: «مساله اصلی در این میان، ابهام مقیاس ذاتی در MDE است. در غیاب اطلاعات هندسی صریح درباره دوربین تصویربرداری، بازیابی عمق مطلق … ذاتا نامعین است.»
    - Change: keep the first sentence, then state the split in one sentence: the part between the
      size of an object and its distance stays in every single-image method and is reduced by what
      the model learns from data; the part that comes from the camera is removed by the calibration
      matrix, and this second part is the subject of the thesis. This is new prose, to be worded in
      Persian when the correction is applied.

15. Glossary — `words.tex` line 13 and both tables in `glossaries-settings.tex` (lines 177 and 267).
    - Add a second entry beside «ابهام مقیاس / Scale Ambiguity»:
      «ابهام مقیاس ناشی از دوربین» / "Metric Ambiguity from Camera Models".

### Leave unchanged (checked, correct as written)

- `chapter1.tex` line 7: «ابهام‌های هندسی متعدد» — general statement.
- `chapter1.tex` line 139: scale ambiguity in 3D reconstruction — about the application.
- `chapter2.tex` line 15: the mapping is not unique; the model must reduce the ambiguity with data,
  scene assumptions or geometric constraints — already correct and complete.
- `chapter2.tex` lines 29, 32, 83: panel الف and the caption of Figure 2-1.
- `chapter2.tex` line 189: scale ambiguity of a relative-depth output — a different, correct use.
- `chapter2.tex` line 265: already says «ابهام مقیاس ناشی از دوربین».
- `chapter3.tex` line 429: «ابهام میان مقیاس صحنه و میدان دید … تنها از طریق اطلاعات ماتریس کالیبراسیون دوربین قابل رفع است» — already names the camera part exactly.
- `chapter4.tex` line 445, `chapter5.tex` lines 17, 90, 91: «ابهام هندسی» or a neutral mention.
- `chapter3.tex` line 366 and `appendix3.tex`: the word is used in its everyday sense.
- The Persian and English abstracts do not use the word.

16. `chapter4.tex` lines 1044–1047 — §«وابستگی سهم پارامترهای دوربین به پروتکل ارزیابی و ظرفیت مدل».
    - Now: «تراز مقیاس و جابجایی به‌ازای هر تصویر، دقیقا همان مولفه اطلاعاتی را از ارزیابی حذف می‌کند که میدان دید و فاصله کانونی در اختیار مدل می‌گذارند؛ یعنی مقیاس مطلق صحنه.»
    - Problem: the field of view and focal length do not give the model the absolute scale of the
      scene; they give the camera part that is needed to recover it (same split as above).
    - Change: say that the alignment removes the component that the field of view and focal length
      help to recover, the absolute scale of the scene. Added 2026-11-20. The defense slide 21
      notes already say «به بازیابی آن کمک می‌کند».

### When applying

- Rebuild and check the figure in chapter 3 (item 1) and the panel title in Figure 2-1 (item 13)
  for overflow; both labels get longer.
- Items 13 and 14 need new Persian sentences; the others are term replacements.
- The defense slides already use the corrected wording (slides 5, 6, 17, 21 and 36 of
  `deliverables/thesis-defense-presentation-v2`).
