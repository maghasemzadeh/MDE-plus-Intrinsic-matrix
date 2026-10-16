"""Build the self-contained SVG example used on defense slide 26."""

from __future__ import annotations

import base64
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thesis/figures/samples/nyu.jpg"
OUT = ROOT / "deliverables/thesis-defense-slides/assets"
PHOTO = base64.b64encode(SOURCE.read_bytes()).decode("ascii")
FONT = "IBM Plex Sans, Arial, sans-serif"
MONO = "IBM Plex Mono, Consolas, monospace"


def card(alpha: float, label: str, accent: str) -> str:
    focal = 500 * alpha
    depth = 2 * alpha
    fov = math.degrees(2 * math.atan(640 / (2 * focal)))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600" role="img" aria-label="{label}: same RGB image, focal length {focal:.0f} pixels, example depth {depth:.1f} metres, horizontal field of view {fov:.1f} degrees">
<rect width="800" height="600" rx="20" fill="#12151d" stroke="#303746" stroke-width="2"/>
<rect x="28" y="24" width="744" height="52" rx="8" fill="{accent}" fill-opacity=".14"/>
<text x="48" y="58" fill="{accent}" font-family="{MONO}" font-size="25" font-weight="600">{label}</text>
<text x="748" y="58" text-anchor="end" fill="#f3f1ea" font-family="{MONO}" font-size="22">α = {alpha:g}</text>
<defs><clipPath id="crop"><rect x="28" y="94" width="744" height="360" rx="10"/></clipPath></defs>
<image x="28" y="-5" width="744" height="558" href="data:image/jpeg;base64,{PHOTO}" clip-path="url(#crop)"/>
<rect x="28" y="94" width="744" height="360" rx="10" fill="none" stroke="#667084" stroke-opacity=".6"/>
<rect x="28" y="472" width="744" height="104" rx="10" fill="#1d2230"/>
<text x="50" y="508" fill="#adb3c2" font-family="{FONT}" font-size="23">focal length</text>
<text x="50" y="546" fill="#f3f1ea" font-family="{MONO}" font-size="29" font-weight="600">{focal:.0f} px</text>
<text x="293" y="508" fill="#adb3c2" font-family="{FONT}" font-size="23">example depth</text>
<text x="293" y="546" fill="#f3f1ea" font-family="{MONO}" font-size="29" font-weight="600">{depth:.1f} m</text>
<text x="543" y="508" fill="#adb3c2" font-family="{FONT}" font-size="23">horizontal FoV</text>
<text x="543" y="546" fill="#f3f1ea" font-family="{MONO}" font-size="29" font-weight="600">{fov:.1f}°</text>
</svg>'''


def comparison() -> str:
    left = card(1.0, "A  /  ORIGINAL", "#5b9df5")
    right = card(1.6, "B  /  FOCAL JITTER", "#a597f4")
    # Reuse the exact same photograph in both states, while keeping each card
    # independently usable as an image asset.
    inner_left = left.split("</svg>")[0].split(">", 1)[1]
    inner_right = right.split("</svg>")[0].split(">", 1)[1]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1760" height="686" viewBox="0 0 1760 686" role="img" aria-label="Two identical RGB images. Alpha 1 uses focal length 500 pixels and depth 2 metres; alpha 1.6 uses focal length 800 pixels and depth 3.2 metres. Pixel projection remains identical.">
<rect width="1760" height="686" rx="18" fill="#0a0c11"/>
<g transform="translate(22 20)">{inner_left}</g>
<g transform="translate(938 20)">{inner_right}</g>
<path d="M 846 270 L 908 270" stroke="#e0a940" stroke-width="5" stroke-linecap="round"/>
<path d="M 902 259 L 916 270 L 902 281" fill="none" stroke="#e0a940" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
<text x="880" y="239" text-anchor="middle" fill="#e0a940" font-family="{MONO}" font-size="29" font-weight="600">×1.6</text>
<text x="880" y="657" text-anchor="middle" fill="#f3f1ea" font-family="{MONO}" font-size="28">same RGB pixels · same projected coordinate: 500 × 0.8 / 2.0 = 800 × 0.8 / 3.2 = 200 px</text>
</svg>'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "focal-jitter-original.svg").write_text(card(1.0, "A  /  ORIGINAL", "#5b9df5"))
    (OUT / "focal-jitter-alpha-1.6.svg").write_text(card(1.6, "B  /  FOCAL JITTER", "#a597f4"))
    (OUT / "focal-jitter-comparison.svg").write_text(comparison())


if __name__ == "__main__":
    main()
