---
name: excalidraw-diagram
description: Create conceptual or workflow diagrams (architecture overviews, pipeline flows, system sketches) as Excalidraw-format files, for diagrams that don't need to compile as part of the LaTeX thesis. Use for repo documentation, planning sketches, or any diagram the user wants editable in the Excalidraw app/UI. For a diagram that must compile into the thesis PDF, use tikz-diagrams instead — don't produce an Excalidraw file for a thesis figure unless explicitly asked for a non-LaTeX deliverable.
---

# Excalidraw Diagrams

Excalidraw files are JSON (`.excalidraw`) describing a scene graph of shapes,
arrows, and text — hand-drawn style, editable visually in the Excalidraw app
or web UI (excalidraw.com, or the VS Code extension). Use this when the
deliverable itself should be an editable diagram, not a compiled figure.

## When to use this vs. the alternatives

- **This skill**: conceptual/workflow diagrams living outside the thesis
  (e.g. a `docs/` architecture sketch, a planning diagram for this CLAUDE.md's
  described pipeline, a diagram meant to be opened and tweaked visually by a
  human afterward).
- **`tikz-diagrams`**: any figure that must appear in `thesis/main.pdf` —
  Excalidraw output is a raster/SVG export, not LaTeX source, so it won't
  inherit the thesis's `thesisplot`/`thblock` color and style conventions and
  shouldn't be used for thesis figures unless the user explicitly wants a
  non-LaTeX diagram embedded as an image.
- **`data-visualization`**: data-driven charts (bars, lines, scatter) from
  real numbers — Excalidraw is for conceptual boxes-and-arrows, not plotting
  data.

## File format basics

An `.excalidraw` file is JSON with this shape:

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "https://excalidraw.com",
  "elements": [
    {
      "type": "rectangle",
      "id": "node1",
      "x": 100, "y": 100, "width": 180, "height": 60,
      "strokeColor": "#1e1e1e", "backgroundColor": "#a5d8ff",
      "fillStyle": "solid", "roughness": 1, "seed": 1
    },
    {
      "type": "text",
      "id": "label1",
      "x": 120, "y": 118, "width": 140, "height": 25,
      "text": "Input Image",
      "fontSize": 16, "fontFamily": 1
    },
    {
      "type": "arrow",
      "id": "arrow1",
      "x": 280, "y": 130, "width": 100, "height": 0,
      "points": [[0, 0], [100, 0]],
      "startBinding": {"elementId": "node1", "focus": 0, "gap": 4},
      "endBinding": {"elementId": "node2", "focus": 0, "gap": 4}
    }
  ],
  "appState": {"viewBackgroundColor": "#ffffff"}
}
```

Every element needs a unique `id`. Arrows bound to shapes (`startBinding`/
`endBinding`) stay attached when a human later drags a box in the app — prefer
binding over free-floating arrow coordinates whenever the arrow connects two
named boxes.

## Building a diagram

1. **List the nodes and edges first in plain text** before writing JSON — get
   the structure (what connects to what, what's a decision vs. a process vs. a
   data artifact) agreed before laying out coordinates.
2. **Lay out left-to-right or top-to-bottom** for a pipeline/flow; grid
   coordinates in increments of ~20-40px so elements align cleanly. Leave at
   least 80-100px between connected boxes for the arrow and its label.
3. **Use shape semantically, consistently**: e.g. rectangles for
   processes/stages, rounded rectangles or ellipses for start/end states,
   diamonds for decisions, a distinct fill color for data/artifact boxes vs.
   process boxes — and hold that convention for every box in the same diagram.
4. **Keep text labels inside their shape's bounds** — a text element wider
   than its parent shape will visually overflow when rendered.
5. **Don't invent structure that wasn't asked for.** If the user describes a
   3-stage pipeline, draw 3 stages — don't add extra decision branches or
   sub-steps not mentioned, since that misrepresents the actual system.

## Verification

After writing the file, if a render/export tool is available, render it to
PNG/SVG and visually check: no overlapping boxes, no arrow pointing at empty
space (binding mismatch), no text overflowing its container, consistent
color/shape semantics across the whole diagram. If no renderer is available,
state that the JSON was authored but not visually verified.

## Reporting

State the file path written, the diagram's node/edge count, and whether it was
rendered and visually checked or only authored as JSON.
