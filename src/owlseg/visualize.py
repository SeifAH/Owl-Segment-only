"""Draw the annotated image and build the scrollable object list."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#ffe119",
]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))


def _text_color(bg: str) -> str:
    r, g, b = _hex_to_rgb(bg)
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(image: Image.Image, objects: list[dict], masks: np.ndarray | None, show_masks: bool = True,
         mask_alpha: float = 0.40, label_style: str = "auto") -> Image.Image:
    """label_style: 'full' (#id score), 'number' (#id), 'none', or 'auto' (number if > 30 objects)."""
    if label_style == "auto":
        label_style = "full" if len(objects) <= 30 else "number"
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    if show_masks and masks is not None:
        for obj, m in zip(objects, masks):
            c = np.array(_hex_to_rgb(obj["color"]), np.float32)
            base[m] = base[m] * (1 - mask_alpha) + c * mask_alpha
    out = Image.fromarray(base.clip(0, 255).astype(np.uint8))

    d = ImageDraw.Draw(out)
    w, h = out.size
    thickness = max(2, round(min(w, h) / 300))
    f = font(max(12, round(min(w, h) / 45)))
    for obj in objects:
        x0, y0, x1, y1 = obj["box"]
        color = obj["color"]
        d.rectangle([x0, y0, x1, y1], outline=color, width=thickness)
        if label_style == "none":
            continue
        text = f"#{obj['id']} {obj['score']:.2f}" if label_style == "full" else f"{obj['id']}"
        tx0, ty0, tx1, ty1 = d.textbbox((0, 0), text, font=f)
        tw, th = tx1 - tx0, ty1 - ty0
        pad = max(2, thickness)
        ly = y0 - th - 2 * pad if y0 - th - 2 * pad >= 0 else y0
        lx = min(x0, w - tw - 2 * pad)
        d.rectangle([lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad], fill=color)
        d.text((lx + pad - tx0, ly + pad - ty0), text, fill=_text_color(color), font=f)
    return out


LIST_CSS = """
<style>
.ss-list{height:640px;overflow-y:auto;border:1px solid var(--border-color-primary,#ddd);border-radius:8px;
  padding:12px;font-family:system-ui,sans-serif;font-size:14px;background:var(--block-background-fill,#fff)}
.ss-total{font-size:20px;font-weight:700;margin:0 0 8px}
.ss-h{font-weight:600;margin:12px 0 4px;opacity:.75;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.ss-row{display:flex;align-items:center;gap:8px;padding:4px 2px;border-bottom:1px solid rgba(128,128,128,.15)}
.ss-sw{width:12px;height:12px;border-radius:3px;flex:none}
.ss-id{width:36px;opacity:.6;font-variant-numeric:tabular-nums}
.ss-lbl{flex:1}
.ss-num{font-variant-numeric:tabular-nums;opacity:.8}
</style>
"""


def build_list_html(objects: list[dict], image_area: int) -> str:
    parts = [LIST_CSS, '<div class="ss-list">', f'<div class="ss-total">Total objects detected: {len(objects)}</div>']
    if objects:
        parts.append('<div class="ss-h">All objects (size · score)</div>')
        for o in objects:
            size = 100 * o["mask_area"] / image_area
            parts.append(
                f'<div class="ss-row"><span class="ss-id">#{o["id"]}</span>'
                f'<span class="ss-sw" style="background:{o["color"]}"></span>'
                f'<span class="ss-lbl">object</span>'
                f'<span class="ss-num">{size:.1f}% · {o["score"]:.2f}</span></div>'
            )
    else:
        parts.append("<p>No objects found. Try a lower objectness threshold.</p>")
    parts.append("</div>")
    return "".join(parts)
