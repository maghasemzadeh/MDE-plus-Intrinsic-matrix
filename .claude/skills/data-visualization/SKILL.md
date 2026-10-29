---
name: data-visualization
description: Create data-driven charts in Python (matplotlib/seaborn/plotly) for exploratory analysis of experiment results — depth-estimation metric comparisons (AbsRel, RMSE, δ1) across models/datasets, training curves, error distributions — outside the LaTeX thesis. Use for quick plots in outputs/, results/, or notebooks. For a chart that must compile into the thesis PDF as native LaTeX, use tikz-diagrams instead, which already defines this project's thesis color/style conventions.
---

# Data Visualization (Python)

Chart-selection guidance, code patterns, and design principles for charts built
from this project's actual metric outputs (e.g. `compare_models.py`,
`compare_dataset_results.py`, or anything under `outputs/`/`results/`). Adapted
from Anthropic's `data-visualization` skill (`anthropics/knowledge-work-plugins`,
Apache-2.0), retargeted to depth-estimation metric comparisons.

## Chart selection

| Showing | Best chart | Alternative |
|---|---|---|
| Metric trend over training epochs | Line chart | Area chart if cumulative |
| Model comparison on one metric (e.g. AbsRel across models) | Horizontal bar, sorted by value | Grouped bar if comparing 2+ metrics per model |
| Per-dataset breakdown of one model | Grouped/clustered bar | Small multiples (one panel per dataset) |
| Error distribution across test images | Histogram | Box/violin plot to compare across models |
| Two metrics' relationship (e.g. AbsRel vs. δ1) | Scatter plot | Bubble chart if a 3rd variable (e.g. sample count) matters |
| Many models × many datasets | Heatmap (model × dataset grid) | Faceted small multiples |
| Baseline vs. proposed, same metric, many conditions | Slope chart (two connected points) | Grouped bar |

**Avoid:** pie/donut charts for metric comparisons (angles are hard to compare
precisely — exactly the opposite of what a metric table needs); 3D charts
(never, they distort real differences); dual-axis charts for two metrics with
different scales unless both axes are clearly labeled and the reader won't
assume correlation that isn't there.

## Setup and style

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Keep colors consistent with this project's model identity if the result
# will sit next to a thesis figure (see tikz-diagrams' thProposed/thControl/
# thBase palette) — reuse the same hex values here for visual continuity:
COLOR_PROPOSED = '#1F6FB2'   # matches thesis \thProposed
COLOR_CONTROL  = '#C0392B'   # matches thesis \thControl
COLOR_BASE     = '#78838C'   # matches thesis \thBase
PALETTE_CATEGORICAL = [COLOR_PROPOSED, COLOR_CONTROL, COLOR_BASE, '#8172B3', '#937860']
```

## Model comparison bar chart

```python
fig, ax = plt.subplots(figsize=(10, 6))

df_sorted = df.sort_values('abs_rel', ascending=True)  # lower AbsRel is better
colors = [COLOR_PROPOSED if m == 'proposed' else COLOR_CONTROL if m == 'control'
          else COLOR_BASE for m in df_sorted['model']]

bars = ax.barh(df_sorted['model'], df_sorted['abs_rel'], color=colors)
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
            f'{width:.3f}', ha='left', va='center', fontsize=10)

ax.set_title('AbsRel by Model (lower is better)', fontweight='bold')
ax.set_xlabel('AbsRel')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('abs_rel_comparison.png', dpi=150, bbox_inches='tight')
```

## Training curve (line chart)

```python
fig, ax = plt.subplots(figsize=(10, 6))
for label, group in df.groupby('model'):
    color = {'proposed': COLOR_PROPOSED, 'control': COLOR_CONTROL}.get(label, None)
    ax.plot(group['epoch'], group['val_abs_rel'], label=label, linewidth=2, color=color)

ax.set_title('Validation AbsRel over Training', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('AbsRel')
ax.legend(loc='upper right', frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
```

## Model × dataset heatmap

```python
fig, ax = plt.subplots(figsize=(10, 8))
pivot = df.pivot_table(index='model', columns='dataset', values='abs_rel', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd_r',  # reversed: lower AbsRel = better = lighter
            linewidths=0.5, ax=ax, cbar_kws={'label': 'AbsRel'})
ax.set_title('AbsRel by Model and Dataset', fontweight='bold')
plt.tight_layout()
plt.savefig('model_dataset_heatmap.png', dpi=150, bbox_inches='tight')
```

## Design principles

- **Lower-is-better metrics (AbsRel, RMSE)**: sort bars ascending, say
  explicitly in the title or axis label which direction is "better" — don't
  make the reader infer it.
- **Higher-is-better metrics (δ1, δ2, δ3 accuracy)**: sort descending, same
  principle.
- **Bars start at zero.** Truncating the axis to exaggerate a small
  improvement is misleading, even unintentionally — this matters especially
  when the point of the chart is to argue the proposed model is better.
- **Show uncertainty when it exists.** If results come from multiple seeds or
  splits, plot error bars or a distribution (box/violin) rather than a bare
  mean bar.
- **Consistent color per model across every chart in a report** — don't let
  "proposed" be blue in one figure and orange in the next. Reuse the hex
  values above (which mirror the thesis's own palette) for continuity if this
  analysis feeds into a thesis chapter.
- **Colorblind safety**: avoid red/green as the only distinguishing pair;
  the palette above uses blue/red/gray which is the safer convention.
- **Label axes with units**: "AbsRel" alone is fine (it's unitless), but RMSE
  must state the unit (meters) if the dataset's depth is metric.

## Accessibility checklist

- [ ] Chart works without color (line style, direct labels, or pattern fills
      differentiate series)
- [ ] Title states which direction is better, not just the metric name
- [ ] Axes labeled with units where applicable
- [ ] Legend doesn't obscure data
- [ ] Source run/config noted (which `compare_models.py`/`compare_dataset_results.py`
      invocation produced this data) — don't let a chart outlive traceability
      to its source command

## Reporting

State the data source (exact command/file that produced the numbers), the
chart type chosen and why, and the output file path.
