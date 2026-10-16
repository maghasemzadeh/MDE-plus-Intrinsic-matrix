# شکل‌های وابسته به دادهٔ بیرونی

این پوشه بلوک‌های شکلی را نگه می‌دارد که به خروجی اسکریپت‌های پژوهش یا به
فایل‌های تصویری بیرون از مخزن لاتک وابسته‌اند. **هر سه بلوک اکنون ساخته و در
`chapter4.tex` وارد شده‌اند**؛ نام پوشه صرفاً به‌دلیل حفظ مسیرهای موجود
دست‌نخورده مانده است.

| فایل | داده‌ای که لازم دارد | وضعیت |
|---|---|---|
| `qualitative_figures.tex` | `figures/qualitative_depth_maps.pdf` و `figures/curves/scale_ratio_hist.csv` | ساخته‌شده — بخش ۴--۶ |
| `collapse_curves.tex` | `figures/curves/train_avg_loss_per_epoch.csv` و `figures/curves/eval_abs_rel.csv` | ساخته‌شده — بخش ۴--۴ |
| `dataset_samples.tex` | چهار تصویر در `figures/samples/` | ساخته‌شده — بخش ۴--۳--۱ |

## بازتولید شکل کیفی (نیازمند GPU)

روی سرور، با فعال‌بودن venv و از ریشهٔ مخزن. **ترتیب دو مرحله اجباری است**:
شناسهٔ تصویرها پیش از هر استنتاجی قفل می‌شود، تا انتخاب نمونه پس از دیدن
خروجی‌ها قابل تغییر نباشد.

```bash
# مرحلهٔ ۱ — قفل‌کردن نمونه (هیچ مدلی بارگذاری نمی‌شود)
python tools/make_qualitative_figure.py lock \
    --num-images 6 --seed 20260821 \
    --manifest results/qualitative/manifest.json
# بذر ۲۰۲۶۰۸۲۱ و چکیدهٔ 8086cb124e40... در عنوان شکل ثبت شده است

# مرحلهٔ ۲ — رندر دقیقاً همان نمونه‌ها، بدون هیچ متنی درون تصویر
python tools/make_qualitative_figure.py render \
    --manifest results/qualitative/manifest.json \
    --checkpoint proposed=models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_focal/revised_metric_vitl_nyu_intrinsics_lr5e-6_bs2_20260709_214151/best.pth \
    --checkpoint control=models/raw_models/DepthAnythingV2-revised/checkpoints/metric_finetuning_focal/base_metric_vitl_nyu_lr5e-6_bs2_20260709_233640/best.pth \
    --encoder vitl --model-type metric --annotate none \
    --output-dir results/qualitative
```

`--annotate none` تصویر را بدون هیچ نوشته‌ای تولید می‌کند؛ سرستون‌های فارسی در
`qualitative_figures.tex` و توسط خود لاتک حروف‌چینی می‌شوند. کسرهای افقی
سرستون‌ها با `COLUMN_CENTRES` در اسکریپت هم‌خوان‌اند و اگر هندسهٔ اسکریپت عوض
شود باید به‌روزرسانی شوند.

## بازتولید منحنی‌های آموزش (بدون GPU)

CSVها در مخزن نگهداری می‌شوند، پس شکل بدون سرور بازساختنی است. برای استخراج
دوباره از رویدادهای TensorBoard (که کنار هر نقطهٔ وارسی ذخیره شده‌اند، نه در
پوشهٔ `runs/`):

```bash
python tools/tensorboard_to_thesis.py \
    --runs models/raw_models/DepthAnythingV2-revised/checkpoints \
    --run collapsed=basic_finetuning/revised_basic_vitl_nyu_intrinsics_lr5e-6_bs4_20260705_235227 \
    --run collapsed_control=basic_finetuning/base_basic_vitl_nyu_lr5e-6_bs4_20260706_010928 \
    --run fixed=basic_finetuning_disp/revised_basic_vitl_nyu_intrinsics_lr5e-6_bs4_20260706_120823 \
    --run fixed_control=basic_finetuning_disp/base_basic_vitl_nyu_lr5e-6_bs4_20260706_133242 \
    --tag train/avg_loss_per_epoch --tag eval/abs_rel \
    --output-dir thesis/figures/curves
```

## تصاویر نمونهٔ مجموعه‌داده‌ها

هر چهار تصویر با `logs/make_samples.py` مستقیماً از خود مجموعه‌داده‌ها استخراج
شده‌اند (نه از مقالات). مسیر دقیق و مجوز هرکدام در
`figures/samples/provenance.json` ثبت است.
