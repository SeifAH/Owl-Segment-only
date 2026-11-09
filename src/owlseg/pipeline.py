"""OWLv2 + SAM object finder: numbered objects with outlines, no labels.

CLI:  uv run python -m owlseg.pipeline path/to/image.jpg [--objectness 0.1]
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .filters import find_objects
from .visualize import PALETTE, build_list_html, draw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@dataclass
class Settings:
    min_objectness: float = 0.1    # OWLv2 "is this a whole object" threshold; lower finds more, and more junk
    min_area: float = 0.002        # fraction of the image
    max_area: float = 0.25         # fraction of the image; larger regions count as background
    max_overlap: float = 0.5       # skip a mask lying this much on objects already kept (duplicates, parts)
    show_masks: bool = True
    label_style: str = "auto"      # auto | full | number | none


@dataclass
class Result:
    annotated: Image.Image
    list_html: str
    objects: list[dict]
    timings: dict[str, float]

    def to_json(self) -> dict:
        return {"total": len(self.objects), "timings_s": self.timings,
                "objects": [{k: v for k, v in o.items() if k != "color"} for o in self.objects]}

    def save(self, stem: str, out_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{stem}_{datetime.now():%Y%m%d-%H%M%S}"
        png, js = out_dir / f"{tag}_annotated.png", out_dir / f"{tag}.json"
        self.annotated.save(png)
        js.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return png, js


class _Progress:
    """Maps the pipeline's stages onto one 0..1 progress bar.

    Each stage gets a slice proportional to its typical run time. During long single steps (model loading,
    OWLv2) a ticker creeps the bar forward so it never looks frozen; it never moves backwards.
    callback(fraction, description) is e.g. a Gradio progress object; None disables reporting.
    """

    def __init__(self, callback, weights: dict[str, float]):
        self.callback = callback
        self.weights = weights
        total = sum(weights.values()) or 1.0
        self.spans, acc = {}, 0.0
        for stage, weight in weights.items():
            self.spans[stage] = (acc / total, (acc + weight) / total)
            acc += weight
        self._lock = threading.Lock()
        self._last = 0.0
        self._tick_id = 0

    def __call__(self, stage: str, desc: str, frac: float = 0.0, *, _tick: int | None = None) -> None:
        if self.callback is None or stage not in self.spans:
            return
        with self._lock:
            if _tick is None:
                self._tick_id += 1  # any real update stops the running ticker
                if frac < 1.0:
                    self._start_ticker(stage, desc, frac)
            elif _tick != self._tick_id:
                return
            a, b = self.spans[stage]
            value = max(min(a + (b - a) * frac, 1.0), self._last)
            self._last = value
            self.callback(value, desc)

    def _start_ticker(self, stage: str, desc: str, frac: float) -> None:
        tick, expected = self._tick_id, max(self.weights.get(stage, 1), 1)

        def run():
            t0 = time.perf_counter()
            while True:
                time.sleep(0.5)
                if tick != self._tick_id:
                    return
                eased = 0.9 * (1 - math.exp(-(time.perf_counter() - t0) / expected))
                self(stage, desc, frac + (1 - frac) * eased, _tick=tick)

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            self._tick_id += 1


STEP_TEXT = {
    "find": "Step 1/2 · OWLv2: looking for objects{extra}...",
    "embed": "Step 1/2 · SAM: analysing the image...",
    "outline": "Step 2/2 · SAM: outlining each object ({pct}%)...",
}


class OwlSamPipeline:
    """Loads the models on first use and keeps them in memory."""

    def __init__(self):
        self._engine = None

    def engine(self, on_status=None):
        if self._engine is None:
            from .engine import OwlSam
            self._engine = OwlSam(on_status=on_status)
            print(f"[owlseg] backend: {self._engine.backend}")
        return self._engine

    def run(self, image: Image.Image, s: Settings, progress=None) -> Result:
        """progress: optional callback(fraction 0..1, description), e.g. a gradio.Progress."""
        from .engine import intel_gpu_available, models_converted
        parallel = self._engine.parallel if self._engine is not None else intel_gpu_available()
        # weights = typical seconds per stage on a laptop (Core Ultra 7 155U)
        load = 0 if self._engine is not None else (3 if models_converted() else 60)
        weights = {"load": load, "find": 6,
                   "embed": 0 if parallel else 5, "outline": 2 if parallel else 4, "filter": 1}
        report = _Progress(progress, {k: v for k, v in weights.items() if v})
        try:
            return self._run(image, s, report, parallel)
        finally:
            report.stop()

    def _run(self, image: Image.Image, s: Settings, report: _Progress, parallel: bool) -> Result:
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        if self._engine is None:
            report("load", "Loading OWLv2 and SAM (first run only)...")
            t = time.perf_counter()
            self.engine(on_status=lambda msg: report("load", msg))
            timings["load"] = time.perf_counter() - t

        extra = " while SAM analyses the image" if parallel else ""

        def on_step(stage, frac):
            report(stage, STEP_TEXT[stage].format(extra=extra, pct=round(100 * frac)), frac)

        masks, scores, _ = self._engine.run(image, s.min_objectness, report=on_step, timings=timings)

        report("filter", "Removing background and duplicates, drawing...")
        t = time.perf_counter()
        found = find_objects(masks, scores, s.min_area, s.max_area, s.max_overlap)
        band = max(h // 12, 1)  # reading order: row bands top-to-bottom, then left-to-right
        found.sort(key=lambda f: (int(f[1][1]) // band, float(f[1][0])))
        objects = [
            {"id": i + 1, "score": round(min(sc, 1.0), 4), "box": [round(float(v), 1) for v in box],
             "mask_area": int(m.sum()), "color": PALETTE[i % len(PALETTE)]}
            for i, (m, box, sc) in enumerate(found)
        ]
        kept = np.stack([m for m, _, _ in found]) if found else None
        annotated = draw(image, objects, kept, show_masks=s.show_masks, label_style=s.label_style)
        timings["filter+draw"] = time.perf_counter() - t
        timings = {k: round(v, 2) for k, v in timings.items()}
        timings["total"] = round(time.perf_counter() - t_start, 2)
        return Result(annotated, build_list_html(objects, w * h), objects, timings)


def main() -> None:
    ap = argparse.ArgumentParser(description="OWLv2 + SAM object finder (no labels)")
    ap.add_argument("image", type=Path)
    ap.add_argument("--objectness", type=float, default=0.1, help="OWLv2 objectness threshold")
    args = ap.parse_args()
    res = OwlSamPipeline().run(Image.open(args.image), Settings(min_objectness=args.objectness))
    png, js = res.save(args.image.stem)
    print(f"Objects : {len(res.objects)}")
    print("Timings :", res.timings)
    print("Saved   :", png, "\n         ", js)


if __name__ == "__main__":
    main()
