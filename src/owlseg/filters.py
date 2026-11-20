"""Turn the raw OWLv2+SAM masks into a list of objects: drop specks, background and duplicates.

Masks are cropped to their own box (see engine.Candidate), so memory stays small on long belt recordings.
"""
from __future__ import annotations

import numpy as np

from .engine import Candidate


def _inside(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box a's area that lies inside box b (boxes are x0, y0, x1, y1)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return float(w * h) / max(float((a[2] - a[0]) * (a[3] - a[1])), 1e-6)


def find_objects(cands: list[Candidate], image_size: tuple[int, int], ref_size: tuple[int, int],
                 min_area_frac=0.002, max_area_frac=0.25, max_overlap=0.5, max_span_frac=0.85,
                 part_inside=0.9, part_score_ratio=0.4, stitch_iou=0.3) -> list[Candidate]:
    """Return the kept objects, best score first.

    Masks are visited best score first. A mask is skipped when it is a speck or too big for one object, spans
    almost the whole width or height (belt, floor), or lies mostly (> max_overlap) on objects already kept, which
    removes duplicate boxes on the same object and boxes around a part of it. Sizes are relative to `ref_size`,
    the (w, h) the models saw at once: the whole image, or one tile of a long recording.

    A part (cap, sleeve, label) often sits next to its object's mask rather than on it, so the mask test misses
    it. It is also dropped when its box lies at least `part_inside` inside a kept object's box and its score is
    below `part_score_ratio` x that object's score. Tested on 6 belt frames: 0.4 removed only true parts, 0.5
    already removed real items (a paper lying on a bottle).

    Objects cut by a tile edge appear once per tile; their pieces are stitched back together when they overlap
    (IoU >= stitch_iou over the shared area), see stitch().
    """
    w, h = image_size
    rw, rh = ref_size
    ref_area = rw * rh
    claimed = np.zeros((h, w), dtype=bool)  # pixels of objects already kept
    out: list[Candidate] = []
    for c in sorted(cands, key=lambda c: -c.score):
        a = int(c.mask.sum())
        if a < min_area_frac * ref_area or a > max_area_frac * ref_area:
            continue
        x0, y0, x1, y1 = (int(v) for v in c.box)
        if x1 - x0 > max_span_frac * rw or y1 - y0 > max_span_frac * rh:
            continue
        if (c.mask & claimed[y0:y1, x0:x1]).sum() / a > max_overlap:
            continue
        if any(_inside(c.box, k.box) >= part_inside and c.score < part_score_ratio * k.score for k in out):
            continue
        out.append(c)
        claimed[y0:y1, x0:x1] |= c.mask
    return stitch(out, stitch_iou)


def _union(a: Candidate, b: Candidate) -> Candidate:
    box = np.array([min(a.box[0], b.box[0]), min(a.box[1], b.box[1]), max(a.box[2], b.box[2]), max(a.box[3], b.box[3])])
    m = np.zeros((box[3] - box[1], box[2] - box[0]), bool)
    for c in (a, b):
        m[c.box[1] - box[1]:c.box[3] - box[1], c.box[0] - box[0]:c.box[2] - box[0]] |= c.mask
    return Candidate(box, m, max(a.score, b.score), tuple(p and q for p, q in zip(a.cut, b.cut)))


def stitch(objs: list[Candidate], min_iou: float = 0.3) -> list[Candidate]:
    """Join pieces of one object that was cut by tile edges: a piece touching an inner tile edge is merged with a
    kept object whose mask covers the same pixels where both boxes overlap."""
    objs = list(objs)
    merged = True
    while merged:
        merged = False
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                if not (any(a.cut) or any(b.cut)):
                    continue
                x0, y0 = max(a.box[0], b.box[0]), max(a.box[1], b.box[1])
                x1, y1 = min(a.box[2], b.box[2]), min(a.box[3], b.box[3])
                if x0 >= x1 or y0 >= y1:
                    continue
                ma = a.mask[y0 - a.box[1]:y1 - a.box[1], x0 - a.box[0]:x1 - a.box[0]]
                mb = b.mask[y0 - b.box[1]:y1 - b.box[1], x0 - b.box[0]:x1 - b.box[0]]
                union = (ma | mb).sum()
                if union and (ma & mb).sum() / union >= min_iou:
                    objs[i] = _union(a, b)
                    del objs[j]
                    merged = True
                    break
            if merged:
                break
    return objs
