"""Turn the raw OWLv2+SAM masks into a list of objects: drop specks, background and duplicates."""
from __future__ import annotations

import numpy as np


def _inside(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box a's area that lies inside box b (boxes are x0, y0, x1, y1)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return float(w * h) / max(float((a[2] - a[0]) * (a[3] - a[1])), 1e-6)


def find_objects(
    masks: np.ndarray,
    scores: np.ndarray,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.25,
    max_overlap: float = 0.5,
    max_span_frac: float = 0.85,
    part_inside: float = 0.9,
    part_score_ratio: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Return [(mask, xyxy_box, score)] for each kept object.

    Masks are visited best score first. A mask is skipped when it is a speck or too big for one object, spans
    almost the whole image (belt, floor), or lies mostly (> max_overlap) on objects already kept, which
    removes duplicate boxes on the same object and boxes around a part of it.

    A part (cap, sleeve, label) often sits next to its object's mask rather than on it, so the mask test misses
    it. It is also dropped when its box lies at least `part_inside` inside a kept object's box and its score is
    below `part_score_ratio` x that object's score. Tested on 6 belt frames: 0.4 removed only true parts, 0.5
    already removed real items (a paper lying on a bottle).
    """
    if len(masks) == 0:
        return []
    n, h, w = masks.shape
    img_area = h * w
    claimed = np.zeros((h, w), dtype=bool)  # pixels of objects already kept
    out = []
    for i in np.argsort(-scores):
        m = masks[i]
        a = m.sum()
        if a < min_area_frac * img_area or a > max_area_frac * img_area:
            continue
        ys, xs = np.nonzero(m)
        if xs.max() - xs.min() > max_span_frac * w or ys.max() - ys.min() > max_span_frac * h:
            continue
        if (m & claimed).sum() / a > max_overlap:
            continue
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
        if any(_inside(box, kept_box) >= part_inside and scores[i] < part_score_ratio * kept_score
               for _, kept_box, kept_score in out):
            continue
        out.append((m, box, float(scores[i])))
        claimed |= m
    return out
