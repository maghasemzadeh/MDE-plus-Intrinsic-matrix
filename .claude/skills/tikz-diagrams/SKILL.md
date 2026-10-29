---
name: tikz-diagrams
description: Create or edit LaTeX-native charts, plots, and diagrams (TikZ/pgfplots) that compile directly into the thesis — metric comparison charts (AbsRel/RMSE across models or datasets), pipeline/architecture figures, schematics. Use for any figure that should live as compilable LaTeX source under thesis/, so it matches the thesis's fonts, colors, and RTL captions automatically. For conceptual/workflow diagrams that don't need to compile as LaTeX, use excalidraw-diagram instead; for exploratory Python charts outside the thesis, use data-visualization.
---

# TikZ / pgfplots Diagrams

This thesis already defines a complete visual style for charts and diagrams in
`thesis/tex/commands.tex`. **Reuse it — do not define new colors, line styles,
or plot styles for a thesis figure unless the existing style genuinely cannot
express what's needed.** Consistency across all thesis figures is the point:
a reader should never need to re-learn a new color key per chart.

## Existing style — use these, don't reinvent them

Defined in `thesis/tex/commands.tex` (search there before adding anything new):

**Semantic colors** (each model/concept always gets the same color everywhere):
```latex
thProposed   % مدل پیشنهادی — proposed model (with camera calibration matrix)
thControl    % مدل کنترل — control model (without calibration matrix)
thBase       % published baseline model
thIdeal      % reference / ideal geometric behavior
thAccent     % emphasis and annotation
thInk        % line/text color for figures
thPanel, thPanelAlt, thPanelGreen, thPanelGray, thPanelRed  % background panels
```

**pgfplots style** — apply `thesisplot` to every chart axis:
```latex
\begin{axis}[thesisplot, xlabel={...}, ylabel={...}]
  \addplot[thProposed, thick] table {...};
  \addplot[thControl, thick] table {...};
\end{axis}
```
This already sets width, height, grid style, tick/label fonts, and legend
style consistent with every other chart in the thesis — don't override
`width`/`height`/`grid` locally unless the figure genuinely needs a different
aspect ratio (e.g. a wide multi-panel comparison).

**TikZ node/arrow styles** — for pipeline/architecture diagrams:
```latex
thblock    % a process/stage box: rounded corners, thin border, centered text
thdata     % a data/artifact box: filled thPanelGray, rounded corners
thnote     % small annotation text
tharrow    % solid directional arrow
thdashed   % dashed arrow (e.g. for an optional/conditional path), in thAccent
```
Example:
```latex
\begin{tikzpicture}
  \node[thdata] (input) {تصویر ورودی};
  \node[thblock, right=1.5cm of input] (model) {\basemodel};
  \node[thdata, right=1.5cm of model] (output) {نقشه عمق};
  \draw[tharrow] (input) -- (model);
  \draw[tharrow] (model) -- (output);
\end{tikzpicture}
```

## Figures must stay direction-safe (see persian-writing)

- Any node/axis label/legend entry containing Latin text (model names, metric
  names like RMSE/AbsRel) must wrap that span in `\lr{}` if it's embedded in a
  Persian label. Use the thesis's named model macros (`\proposedmodel`,
  `\controlmodel`, `\basemodel`, `\tableproposedmodel`, etc.) for legend/node
  text instead of typing the name inline, so it stays consistent with prose
  and tables elsewhere.
- Captions follow the same RTL rules as body text — load `persian-writing`
  alongside this skill whenever the figure has a Persian caption or in-figure
  Persian labels (almost always true in this thesis).
- Decimal numbers inside a chart's data labels or annotations should use
  `\fadecimal{}` if they're placed as Persian-context text (not needed inside
  `pgfplots` tick labels, which `displaymathdigits=persian` already handles).

## Workflow

1. **Check for an existing figure to extend first.** Search `thesis/tex/*.tex`
   and `thesis/figures/`/`thesis/img/` for a chart already covering a similar
   comparison — extending a `groupplots` panel is usually better than creating
   a near-duplicate figure.
2. **Source real data.** Never invent axis values. Pull numbers from the
   project's actual output/result files (see `src/metrics.py`,
   `compare_models.py`, `compare_dataset_results.py` outputs) and cite the
   exact run/config that produced them in a code comment above the `\addplot`
   data or in the commit message — not in the rendered figure.
3. **Use `pgfplots` table input** (`\addplot table {...}` or
   `\pgfplotstableread`) rather than hardcoding long inline coordinate lists
   when there are more than ~6 data points — keeps the source maintainable and
   the numbers auditable against their source file.
4. **Place the figure** under `thesis/figures/` (vector/TikZ source) or inline
   in the relevant chapter file, following whatever placement convention
   neighboring figures in that chapter already use.
5. **Label and reference**: every figure needs a `\label{fig:...}` and the
   calling text must `\ref{}` it — never leave a figure unreferenced from
   prose, and never leave a `\ref` dangling.
6. **Build and inspect.** Run `thesis/build.sh` and visually check the
   rendered page: pgfplots/TikZ errors often compile "successfully" while
   silently clipping content or misplacing a legend — a build with no LaTeX
   errors is not proof the figure looks right.

## Chart-type discipline (same principles as data-visualization)

- Bar charts start at zero. Never truncate the y-axis to exaggerate a
  difference between models.
- Use consistent axis ranges across panels that are meant to be compared
  side-by-side (e.g. all `groupplots` subplots sharing a metric scale).
- Avoid 3D plots and pie charts in pgfplots — same reasoning as standard chart
  design: TikZ can render them, but they distort comparison for exactly the
  cases this thesis needs (metric comparisons across models/datasets).
- Label every axis with its unit. A bare "AbsRel" or "RMSE" without stating
  it's unitless/in meters invites misreading.

## Reporting

State which existing style elements were reused, which data source each number
came from, the figure's label/file location, and the build/visual-verification
result.
