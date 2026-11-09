"""Turn the raw OWLv2+SAM masks into a list of objects: drop specks, background and duplicates."""
from __future__ import annotations

import numpy as np
import torch


def _inside(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box a's area that lies inside box b (boxes are x0, y0, x1, y1)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return float(w * h) / max(float((a[2] - a[0]) * (a[3] - a[1])), 1e-6)


def _mask_stats(masks: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Area and inclusive bounding box (x0, y0, x1, y1) of every mask, computed at once (on the GPU with CUDA)."""
    n, h, w = masks.shape
    areas = masks.reshape(n, -1).sum(1)
    rows, cols = masks.any(2), masks.any(1)
    ys = torch.arange(h, device=masks.device)
    xs = torch.arange(w, device=masks.device)
    box = torch.stack([torch.where(cols, xs, w).amin(1), torch.where(rows, ys, h).amin(1),
                       torch.where(cols, xs, -1).amax(1), torch.where(rows, ys, -1).amax(1)], 1)
    return areas.cpu().numpy(), box.cpu().numpy()


def find_objects(
    masks: torch.Tensor,
    scores: np.ndarray,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.25,
    max_overlap: float = 0.5,
    max_span_frac: float = 0.85,
    part_inside: float = 0.9,
    part_score_ratio: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray, float, int]]:
    """Return [(mask cropped to its box, xyxy_box, score, area in px)] for each kept object.

    Masks are visited best score first. A mask is skipped when it is a speck or too big for one object, spans
    almost the whole image (belt, floor), or lies mostly (> max_overlap) on objects already kept, which
    removes duplicate boxes on the same object and boxes around a part of it.

    A part (cap, sleeve, label) often sits next to its object's mask rather than on it, so the mask test misses
    it. It is also dropped when its box lies at least `part_inside` inside a kept object's box and its score is
    below `part_score_ratio` x that object's score. Tested on 6 belt frames: 0.4 removed only true parts, 0.5
    already removed real items (a paper lying on a bottle).

    `masks` is a (N, H, W) bool tensor on any device; only the kept masks, cropped to their boxes, come back.
    A mask is zero outside its box, so every test on the crop gives the same answer as on the full mask.
    """
    if len(masks) == 0:
        return []
    n, h, w = masks.shape
    img_area = h * w
    areas, bounds = _mask_stats(masks)
    claimed = torch.zeros((h, w), dtype=torch.bool, device=masks.device)  # pixels of objects already kept
    out = []
    for i in np.argsort(-scores):
        a = int(areas[i])
        if a < min_area_frac * img_area or a > max_area_frac * img_area:
            continue
        x0, y0, x1, y1 = (int(v) for v in bounds[i])
        if x1 - x0 > max_span_frac * w or y1 - y0 > max_span_frac * h:
            continue
        m = masks[i, y0:y1 + 1, x0:x1 + 1]
        c = claimed[y0:y1 + 1, x0:x1 + 1]
        if int((m & c).sum()) / a > max_overlap:
            continue
        box = np.array([x0, y0, x1 + 1, y1 + 1], np.float32)
        if any(_inside(box, kept_box) >= part_inside and scores[i] < part_score_ratio * kept_score
               for _, kept_box, kept_score, _ in out):
            continue
        c |= m
        out.append((m, box, float(scores[i]), a))
    return [(m.cpu().numpy(), box, score, a) for m, box, score, a in out]
