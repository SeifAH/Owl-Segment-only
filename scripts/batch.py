"""Run the OWLv2 + SAM object finder on every image in a folder; write annotated images + CSV summaries.

    uv run python scripts/batch.py <folder> [--out DIR] [--objectness 0.1]

Output (default: outputs/batch_<timestamp>/):
    annotated/<image>.png    annotated image per input
    summary.csv              one row per image: object count, timing
    objects.csv              one row per object: image, id, score, box, mask area (px and % of image)
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from owlseg.pipeline import OUTPUT_DIR, OwlSamPipeline, Settings

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-run the OWLv2 + SAM object finder on a folder of images")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output folder (default outputs/batch_<timestamp>)")
    ap.add_argument("--objectness", type=float, default=0.1, help="OWLv2 objectness threshold")
    args = ap.parse_args()

    folder = args.folder.expanduser().resolve()
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT) if folder.is_dir() else []
    if not images:
        raise SystemExit(f"No images ({', '.join(sorted(IMAGE_EXT))}) found in {folder}")

    out = (args.out or OUTPUT_DIR / f"batch_{datetime.now():%Y%m%d-%H%M%S}").resolve()
    (out / "annotated").mkdir(parents=True, exist_ok=True)
    s = Settings(min_objectness=args.objectness)
    pipe = OwlSamPipeline()

    print(f"{len(images)} images from {folder}\nResults -> {out}\n", flush=True)
    rows, obj_rows, failed = [], [], []
    t_all = time.perf_counter()
    for n, path in enumerate(images, 1):
        try:
            img = Image.open(path)
            res = pipe.run(img, s)
        except Exception as e:  # keep going; report at the end
            print(f"[{n}/{len(images)}] {path.name}: FAILED ({e})", flush=True)
            failed.append(path.name)
            continue
        res.annotated.save(out / "annotated" / f"{path.stem}.png")
        area = res.annotated.width * res.annotated.height
        rows.append({"image": path.name, "objects": len(res.objects), "time_s": res.timings["total"]})
        for o in res.objects:
            x0, y0, x1, y1 = o["box"]
            obj_rows.append({"image": path.name, "id": o["id"], "score": o["score"], "x0": x0, "y0": y0,
                             "x1": x1, "y1": y1, "mask_area_px": o["mask_area"],
                             "mask_area_pct": round(100 * o["mask_area"] / area, 3)})
        print(f"[{n}/{len(images)}] {path.name}: {len(res.objects)} objects in {res.timings['total']} s", flush=True)

    with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=["image", "objects", "time_s"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "objects.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "id", "score", "x0", "y0", "x1", "y1", "mask_area_px",
                                          "mask_area_pct"])
        w.writeheader()
        w.writerows(obj_rows)

    total = sum(r["objects"] for r in rows)
    print(f"\nDone: {len(rows)} images, {total} objects in {time.perf_counter() - t_all:.0f} s.")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")
    print(f"Summary: {out / 'summary.csv'}\nPer object: {out / 'objects.csv'}\nImages: {out / 'annotated'}")


if __name__ == "__main__":
    main()
