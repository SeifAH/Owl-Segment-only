"""Draw the annotated image: selected objects highlighted in light blue (no boxes), optionally numbered.

draw() bakes the highlights into one image (saved PNG, batch). viewer_html() builds the app's image with one
separate highlight layer per object, so ticking an object in the app only shows or hides its layer.
"""
from __future__ import annotations

import base64
import io
from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HIGHLIGHT = (79, 195, 247)   # light blue fill
OUTLINE = (2, 119, 189)      # darker blue edge, keeps touching objects apart
HIGHLIGHT_ALPHA = 0.5


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _edge(m: np.ndarray, width: int) -> np.ndarray:
    """Pixels of mask m within `width` px of its border (m minus its erosion)."""
    inner = m.copy()
    for _ in range(width):
        shrunk = inner.copy()
        shrunk[1:, :] &= inner[:-1, :]
        shrunk[:-1, :] &= inner[1:, :]
        shrunk[:, 1:] &= inner[:, :-1]
        shrunk[:, :-1] &= inner[:, 1:]
        inner = shrunk
    return m & ~inner


def draw(image: Image.Image, objects: list[dict], masks: np.ndarray | None,
         selected: Iterable[int] | None = None, label_style: str = "auto") -> Image.Image:
    """Highlight the objects whose id is in `selected` (None = all) in light blue.

    label_style: 'full' (#id score), 'number' (#id), 'none', or 'auto' (number if > 30 objects).
    Labels are drawn only on highlighted objects, centred on the object.
    """
    if label_style == "auto":
        label_style = "full" if len(objects) <= 30 else "number"
    sel = {o["id"] for o in objects} if selected is None else set(selected)
    shown = [(o, m) for o, m in zip(objects, masks if masks is not None else []) if o["id"] in sel]

    base = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w = base.shape[:2]
    edge_w = max(1, round(min(w, h) / 500))
    fill = np.array(HIGHLIGHT, np.float32)
    for _, m in shown:
        base[m] = base[m] * (1 - HIGHLIGHT_ALPHA) + fill * HIGHLIGHT_ALPHA
    for _, m in shown:
        base[_edge(m, edge_w)] = OUTLINE
    out = Image.fromarray(base.clip(0, 255).astype(np.uint8))

    if label_style != "none" and shown:
        d = ImageDraw.Draw(out)
        f = font(max(12, round(min(w, h) / 50)))
        stroke = max(1, round(min(w, h) / 600))
        for o, _ in shown:
            x0, y0, x1, y1 = o["box"]
            text = f"#{o['id']} {o['score']:.2f}" if label_style == "full" else f"{o['id']}"
            d.text(((x0 + x1) / 2, (y0 + y1) / 2), text, font=f, anchor="mm", fill="#ffffff",
                   stroke_width=stroke, stroke_fill="#01304d")
    return out


def _data_uri(img: Image.Image, fmt: str, **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


def viewer_html(image: Image.Image, objects: list[dict], masks: np.ndarray | None,
                selected: Iterable[int] | None = None, label_style: str = "auto") -> str:
    """The image with one ready-made transparent highlight layer per object; only `selected` (None = all) shown.

    Each layer is a small RGBA PNG cropped to its object and placed over the image in % units, so the
    browser can show or hide it instantly (see app.py) without asking Python to redraw anything.
    """
    if label_style == "auto":
        label_style = "full" if len(objects) <= 30 else "number"
    sel = None if selected is None else set(selected)
    w, h = image.size
    edge_w = max(1, round(min(w, h) / 500))
    f = font(max(12, round(min(w, h) / 50)))
    stroke = max(1, round(min(w, h) / 600))
    pad = f.size if hasattr(f, "size") else 12  # room around the object so its label is not cut off
    layers = []
    for o, m in zip(objects, masks if masks is not None else []):
        bx0, by0, bx1, by1 = o["box"]
        x0, y0 = max(int(bx0) - pad, 0), max(int(by0) - pad, 0)
        x1, y1 = min(int(bx1) + pad, w), min(int(by1) + pad, h)
        crop = m[y0:y1, x0:x1]
        rgba = np.zeros((*crop.shape, 4), np.uint8)
        rgba[crop] = (*HIGHLIGHT, round(255 * HIGHLIGHT_ALPHA))
        rgba[_edge(crop, edge_w)] = (*OUTLINE, 255)
        layer = Image.fromarray(rgba, "RGBA")
        if label_style != "none":
            text = f"#{o['id']} {o['score']:.2f}" if label_style == "full" else f"{o['id']}"
            ImageDraw.Draw(layer).text(((bx0 + bx1) / 2 - x0, (by0 + by1) / 2 - y0), text, font=f, anchor="mm",
                                       fill="#ffffff", stroke_width=stroke, stroke_fill="#01304d")
        pos = (f"left:{100 * x0 / w:.4f}%;top:{100 * y0 / h:.4f}%;"
               f"width:{100 * (x1 - x0) / w:.4f}%;height:{100 * (y1 - y0) / h:.4f}%"
               + ("" if sel is None or o["id"] in sel else ";display:none"))
        layers.append(f'<img class="ov" data-id="{o["id"]}" style="{pos}" src="{_data_uri(layer, "PNG")}">')
    base = _data_uri(image.convert("RGB"), "JPEG", quality=90)
    return f'<div id="ov-wrap"><div class="ov-box"><img class="ov-base" src="{base}">{"".join(layers)}</div></div>'


def object_label(o: dict, image_area: int) -> str:
    """One line in the object list: '#id  object · size% · score'."""
    return f"#{o['id']}  object · {100 * o['mask_area'] / image_area:.1f}% · {o['score']:.2f}"
