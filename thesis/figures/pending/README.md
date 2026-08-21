# شکل‌های در انتظار دادهٔ بیرونی

بلوک‌های آمادهٔ این پوشه عمداً در `main.tex` وارد **نشده‌اند**، چون به فایل‌های
تصویری‌ای اشاره می‌کنند که هنوز ساخته نشده‌اند؛ اگر همین حالا `\input` شوند،
کامپایل به‌دلیل نبودِ فایل تصویر شکست می‌خورد.

هر بلوک، بالای خودش دقیقاً می‌گوید چه فایلی لازم دارد و کجا باید `\input` شود.

| فایل | چه چیزی لازم دارد | چگونه ساخته می‌شود |
|---|---|---|
| `qualitative_figures.tex` | `figures/qualitative_depth_maps.pdf` و `figures/scale_ratio_distribution.pdf` | `tools/make_qualitative_figure.py` روی سرور (نیازمند GPU) |
| `dataset_samples.tex` | چهار تصویر در `figures/samples/` | تصاویر نمونهٔ مجموعه‌داده‌ها |

## دستور ساخت شکل کیفی

روی سرور، با فعال‌بودن venv و از ریشهٔ مخزن. **ترتیب دو مرحله اجباری است**:
شناسهٔ تصویرها پیش از هر استنتاجی قفل می‌شود، تا انتخاب نمونه پس از دیدن
خروجی‌ها قابل تغییر نباشد.

```bash
# مرحلهٔ ۱ — قفل‌کردن نمونه (هیچ مدلی بارگذاری نمی‌شود)
python tools/make_qualitative_figure.py lock \
    --num-images 6 --seed 20260821 \
    --manifest results/qualitative/manifest.json
# چکیدهٔ SHA-256 چاپ‌شده را یادداشت کنید و در عنوان شکل بیاورید

# مرحلهٔ ۲ — رندر دقیقاً همان نمونه‌ها
python tools/make_qualitative_figure.py render \
    --manifest results/qualitative/manifest.json \
    --checkpoint proposed=<مسیر نقطهٔ وارسی مدل پیشنهادی دور ۳>/best.pth \
    --checkpoint control=<مسیر نقطهٔ وارسی مدل کنترل دور ۳>/best.pth \
    --encoder vitl --model-type metric \
    --output-dir results/qualitative
```

سپس دو فایل PDF خروجی را در `thesis/figures/` کپی کنید و
`qualitative_figures.tex` را در انتهای بخش ۴--۶ `\input` نمایید.

## نمودار منحنی‌های آموزش

```bash
python tools/tensorboard_to_thesis.py --runs runs/ --list      # ابتدا فهرست
python tools/tensorboard_to_thesis.py --runs runs/ \
    --run collapsed=<نام اجرای فروریخته> \
    --run fixed=<نام اجرای اصلاح‌شده> \
    --tag train/loss --tag val/absrel \
    --output-dir thesis/figures/curves
```

خروجی، یک CSV و یک بلوک `pgfplots` آمادهٔ `\input` برای هر برچسب است. چون
CSVها فایل متنی کوچک‌اند، می‌توان آن‌ها را در مخزن نگه داشت و شکل را بدون
دسترسی به سرور بازسازی کرد.
