from pathlib import Path
import re, html
p=Path('deliverables/thesis-defense-slides/camera-intrinsics-thesis-defense.html')
s=p.read_text()
def notes(title, content):
 global s
 pattern=r'(<section\b[^>]*data-title="'+re.escape(title)+r'"[^>]*data-notes=")[^"]*(")'
 s,n=re.subn(pattern,lambda m:m[1]+html.escape(content,quote=True)+m[2],s,count=1)
 assert n==1,title

def add_notes(title, content):
 global s
 pattern=r'(<section\b[^>]*data-title="'+re.escape(title)+r'"[^>]*data-notes=")([^"]*)(")'
 s,n=re.subn(pattern,lambda m:m[1]+m[2]+' '+html.escape(content,quote=True)+m[3],s,count=1)
 assert n==1,title

# Correct the complete crop + resize transformation, matching _geometric_aug.
s=s.replace('c′x=cx−x0, c′y=cy−y0', 'c′x=(cx−x0)·Wnet/wwin, c′y=(cy−y0)·Hnet/hwin')
s=s.replace("cx'=cx-x0, cy'=cy-y0", "cx'=(cx-x0)*Wnet/wwin, cy'=(cy-y0)*Hnet/hwin")
s=s.replace('<i>c<sub>x</sub>&minus;x<sub>0</sub></i>,&nbsp;&nbsp;<i>c<sub>y</sub>&prime;</i><span class="op">=</span><i>c<sub>y</sub>&minus;y<sub>0</sub></i>', '<i>(c<sub>x</sub>&minus;x<sub>0</sub>)W<sub>net</sub>/w<sub>win</sub></i>,&nbsp;<i>c<sub>y</sub>&prime;</i><span class="op">=</span><i>(c<sub>y</sub>&minus;y<sub>0</sub>)H<sub>net</sub>/h<sub>win</sub></i>')
# Show MLP depth and additive CLS route explicitly.
s=s.replace('font-size="11.5">MLP</text>', 'font-size="11.5">2-layer MLP</text>')
s=s.replace('<text x="350" y="225" text-anchor="middle" font-size="11" opacity="0.55">add g·camera token to CLS at configurable layer (default: first)</text>', '<text x="350" y="115" text-anchor="middle" font-size="10.5" opacity="0.75">CLS ← CLS + g·camera token</text>')
s=s.replace('<line x1="530" y1="279" x2="235" y2="176" stroke="#a597f4" stroke-width="1.4" stroke-dasharray="3 3" marker-end="url(#ah2)"/>','<polyline points="530,279 530,255 235,255 235,176" fill="none" stroke="#a597f4" stroke-width="1.4" marker-end="url(#ah2)"/>')
# No numerical threshold smaller than the actual maximum p (2.527e-24).
s=s.replace('Every metric, decisively — <i>p</i> &le; 10<sup>&minus;24</sup>', 'All five metrics improve after Holm correction')
s=s.replace('p ≤ 10⁻²⁴', 'Holm p ≤ 2.53×10⁻²⁴')
s=s.replace('all five metrics · Holm-corrected ·', 'all five metrics · Holm-corrected ·')
# Fix actual exponent typos in HTML rows.
s=s.replace('4.0&times;10&#8315;&sup3;&sup9;', '4.0&times;10<sup>−31</sup>')
s=s.replace('1.2&times;10&#8315;&#8308;&#8304;', '1.2&times;10<sup>−39</sup>')
# Preserve exact narrative from the clarified user brief.
add_notes('Bug #2 — ViT-L Dead-ReLU Collapse', 'روایت دفاع: وقتی از ViT-S به ViT-L رفتیم، هر دو بازوی کنترل و پیشنهادی هم‌زمان خراب شدند، در حالی که نمودار loss طبیعی بود. AbsRel اعتبارسنجی روی 0.346 ثابت ماند. کاهش head learning-rate multiplier از 10 به 1 و اضافه کردن warm-up مشکل را حل نکرد. بررسی مرحله‌به‌مرحله نشان داد سهم پیکسل‌های صفر در 18 گام از 0.001 به 1.0 رسید و خروجی در همه‌جا صفر شد. مدل ازپیش‌آموزش‌دیده disparity می‌داد: نقاط نزدیک مقدار بزرگ‌تری دارند؛ هدف آموزشی قبلی depth بود و همین جهت و دامنهٔ ناسازگار مشکل ایجاد کرد. ViT-S ظاهراً با بازآموزی head دوام آورده بود؛ این را به‌عنوان تفسیر تشخیصی، نه مزیت اثبات‌شدهٔ معماری کوچک، بیان کنید. با هدف 1/max(D_GT,d_min)، اعتبارسنجی در فضای inverse depth و ثبت basic_target_space=disparity قرارداد خروجی اصلاح شد. بازتولید 150 گامی پس از اصلاح: loss حدود 20–50 برابر کمتر و بدون پیکسل مرده. منبع تاریخی: git a51e35f:REPORT_INTRINSICS_EXPERIMENT.md؛ کد فعلی train.py و loss.py.')
add_notes('Negative Result — Zero-Shot Fine-Tuning Fails', 'Why the dataset mixture was problematic: different visual domains, depth ranges, cameras, ground-truth quality, and label conventions. The observed fine-tuning degradation is consistent with adaptation toward this small mixture; the separate contribution of each factor was not ablated. DA2 training uses 595K synthetic labelled images and 62M+ real unlabelled images converted to pseudo-labelled supervision (official source: https://depth-anything-v2.github.io/). روایت دفاع: به‌جای افزایش تنوع با مجموعه‌های کوچک و ناهمگون، برای مطالعهٔ کنترل‌شده از NYUv2 و تنوع هندسی با K صحیح استفاده کردیم.')
add_notes('Training Strategy — Matched Pairs', 'Protocol precision: the optimizer base/camera learning rate is 5×10⁻⁶; the DPT head uses 10× this rate, i.e. 5×10⁻⁵. Only trainable parameters are updated; DINOv2 stays frozen. Full seed-locking of sample order and augmentation draws is explicitly verified for the later focal-jitter pairs; earlier pairs are matched by their recorded recipe, without a multi-seed variance estimate. Knowledge distillation was disabled for the final controlled study.')
add_notes('Bug #1 — NYU Test-Set Leakage', 'تقسیم قبلی روی همهٔ 1,449 جفت برچسب‌دار به‌صورت تصادفی 80/20 انجام می‌شد. راه‌حل، کنار گذاشتن اندیس‌های رسمی Eigen-654 پیش از هر تقسیم است؛ سپس فقط 795 نمونهٔ باقی‌مانده 80/20 می‌شوند: 636 train، 159 validation، 654 test. این اصلاح به split استاندارد مربوط است و به معنی تقسیم بر حسب کلاس یا مقدار برچسب عمق نیست. نتیجهٔ ViT-S بعدی فقط پس از این اصلاح قابل استناد است.')
add_notes('Focal-Jitter — Making Scale Unrecoverable Without K', 'Training precedes the K-sweep diagnostic: α is drawn log-uniformly in [0.625,1.6], and both focal lengths and depth labels are multiplied by α while RGB and principal point stay unchanged. This deliberately introduces label-scale ambiguity for the K-blind arm. Its loss-optimal compromise depends on the loss, so it need not equal the arithmetic mean of α. K supplies information to resolve the imposed factor within this controlled setting; this does not prove K alone resolves every monocular scene ambiguity. Training uses m=1.6, max_depth=20 m (10×1.6=16 m avoids clipping), SiLog λ=0.3, bs=2, accumulation=2, seed=42. The validity-mask ceiling is scaled with α. Same pixels with different focal factors at inference are then the K-sweep test; this observed response was not the cause of initiating training.')
s=s.replace('The best it can do is regress to the mean of the &alpha; distribution', 'Its best prediction is a loss-dependent compromise over the unresolved &alpha; distribution')
s=s.replace("The control's error is exactly the &alpha; noise it can't resolve", 'The camera-aware model has a tighter scale-ratio distribution')
s=s.replace('it is doing exactly what a model regressing to the mean of an unresolvable random variable should do.', 'this is consistent with residual scale bias after ambiguous training.')
s=s.replace('is direct visual evidence that reading', 'is descriptive evidence consistent with reading')
s=s.replace('The model needs camera intrinsics — especially focal length — alongside the image.', 'Camera intrinsics supply an additional focal-length signal alongside the image.')
# Keep raw and corrected statistical readings together.
s=s.replace('AbsRel p=0.040 (Wilcoxon). <b', 'AbsRel raw p=0.040; Holm p=0.1595 (n.s.). <b')
s=s.replace('from marginal (<i style="font-family:var(--serif)">p</i>=0.04) to overwhelming.', 'to a strong within-checkpoint difference that survives Holm correction.')
s=s.replace('p=0.040) even where RMSE was decisive', 'raw p=0.040; Holm p=0.1595, n.s.) even where RMSE remained significant')
# Retain knockout as controlled causal dependence; geometry is approximate, not uniquely identified.
s=s.replace('The degradation matches geometry, quantitatively', 'The scale response is consistent with geometry')
s=s.replace('<b>Not just "wrong K hurts" &mdash; hurts by the amount geometry says it should.</b>', '<b>This is approximate consistency, not an exact prediction of total AbsRel.</b>')
s=s.replace('monotonic, quantitatively predicted degradation', 'controlled input intervention: calibration errors degrade accuracy')
s=s.replace('wrong K degrades by the amount geometry predicts', 'wrong K degrades accuracy; scale response is consistent with geometry')
add_notes('K-Knockout — Causal Proof, Not Just Correlation', 'Scope correction: |X^s−1| predicts a pure multiplicative scale error under ideal assumptions, not the full observed AbsRel with spatially varying baseline errors. The results establish causal dependence on the K input for this checkpoint; they do not uniquely prove that the network implements the pinhole equation. The 59% figure is an increase in error, not a 59% loss of accuracy. Raw per-image knockout arrays were not available locally for independent recomputation; the table is corroborated by the thesis and historical experiment report.')
# Explain that a p-value trajectory is not an effect-size trajectory; show one test family consistently.
s=s.replace('The effect scales with geometric ambiguity — replicated four times', 'The matched RMSE gap grows across these training settings')
s=s.replace('RMSE Wilcoxon <i style="font-family:var(--serif)">p</i>-value (log scale) vs. training round. Each round is an independently-trained matched pair.', 'RMSE p-values: paired t-test in rounds 1–2; Wilcoxon in rounds 3–4. Different recipes are not independent-seed replications.')
notes('The Dose-Response Pattern, Across Four Rounds', 'Across separately trained recipes, control minus proposed RMSE is 0.003654555 (round 1), 0.005446583 (round 2), 0.036073617 (focal-jitter λ=0.3), and 0.0510 (focal-jitter λ=0.5, thesis-reported). Relative RMSE reductions are 1.25%, 1.82%, 10.51%, and 14.58%. The plotted p-values are paired-t 0.0049 and 6.468e-5 for rounds 1–2, then Wilcoxon 1.010e-24 and 7.5e-40 for rounds 3–4. A p-value is not an effect size; the increase in the actual RMSE gap is the relevant trend. Round 4 changes λ while retaining focal jitter, so it is not a fourth level of FoV diversity. Each recipe has one seed per arm; this is evidence across settings, not repeated-seed replication.')
# Historical ablation/source status, keep values but avoid false fresh-validation claim.
add_notes('Ablation — Isolating the Cause', 'Validation scope: these round-4 means, Wilcoxon tests and 0.97±0.03 slope are corroborated by the thesis and historical experiment report, but the local fresh-inference archive covers rounds 1–3 only. Treat this as a supporting ablation, not an independently recomputed fourth seed. The between-λ score difference is descriptive; it does not establish that λ=0.5 is generally optimal.')
s=s.replace('A fourth independent replication of the mechanistic result.', 'Supporting λ=0.5 ablation; one pair, historical report.')
s=s.replace('Effect depends on encoder capacity', 'Observed K benefit differs between ViT-S and ViT-L')
s=s.replace('which extracts more scale info from content alone', 'capacity explanation is a hypothesis')
# Correct chart axis labels to the actual y geometry (0,10,20,30,40).
s=s.replace('y="184" text-anchor="end">10&#8315;&#8313;', 'y="185" text-anchor="end">10<sup>−10</sup>')
# SVG does not support HTML sup; replace the entire labels and grid positions.
a=s.index('<!-- y gridlines: 0,10,20,30,40')
b=s.index('<!-- significance threshold',a)
s=s[:a]+'''<!-- ordinate: -log10(p), with regular 10-unit ticks -->
                <g opacity="0.7">
                  <text x="80" y="244" text-anchor="end">0</text>
                  <text x="80" y="189" text-anchor="end">10</text>
                  <text x="80" y="134" text-anchor="end">20</text>
                  <text x="80" y="79" text-anchor="end">30</text>
                  <text x="80" y="24" text-anchor="end">40</text>
                </g>
                <line x1="90" y1="185" x2="960" y2="185" stroke="var(--line)"/>
                <line x1="90" y1="130" x2="960" y2="130" stroke="var(--line)"/>
                <line x1="90" y1="75" x2="960" y2="75" stroke="var(--line)"/>
                '''+s[b:]
# Avoid claim that relative-protocol test alone determines K irrelevance.
s=s.replace('the camera token has nothing left to add under this protocol.', 'this protocol removes global scale; the observed K benefit is small in this setting.')
# All slide footers should carry their own number, including PDF exports.
s=s.replace("document.querySelectorAll('.slide .slide-count').forEach(function(el){ el.textContent = (idx+1)+' / '+total; });", "slides.forEach(function(sl, i){ var el=sl.querySelector('.slide-count'); if(el) el.textContent=(i+1)+' / '+total; });")
p.write_text(s)
print('updated content and exact scope')
