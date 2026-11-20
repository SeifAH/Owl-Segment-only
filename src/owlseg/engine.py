"""OWLv2 finds the objects (boxes), SAM outlines each one (masks). No class names, no labels.

OWLv2 is an open-vocabulary detector; here only its image half runs. For each of its 3600 image patches
(60 x 60 at 960 px) it predicts a box and an *objectness* score: "how likely is there a whole object here",
independent of any class name. Boxes above `min_objectness` survive non-maximum suppression and are passed to
SAM as box prompts; SAM returns one mask per box.

Images larger than TILE px (long line-scan belt recordings) run tile by tile, so neither model has to shrink
them by 10-40x; see run() and docs/HOW_IT_WORKS.md.

Where each model runs (chosen automatically, fastest first):
  * OWLv2 on the CPU and SAM on the Intel integrated GPU, both via OpenVINO in fp32, run at the same time
    (~8 s per image on a Core Ultra 7 155U).
  * No Intel GPU: everything via OpenVINO on the CPU, one after the other.
  * OpenVINO fails: PyTorch on the CPU (slowest, same results).
Set OWLSEG_DEVICE=cpu to skip the GPU, OWLSEG_BACKEND=torch to force PyTorch.

fp32 everywhere is deliberate: the GPU's default fp16 mode found fewer objects in tests (78% vs 81%).
Converted models are saved once to OV_CACHE (first start ~1 minute); compiled GPU kernels are cached there too.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms

from .models import OWL_MODEL, SAM_MODEL

BOX_BATCH = 32  # SAM box prompts per decoder call (static shape for OpenVINO; last batch is padded)
OWL_SIZE = 960  # OWLv2-base input resolution
TILE = 2048  # images larger than this are processed in tiles of TILE x TILE px ...
TILE_OVERLAP = 512  # ... overlapping by this much; objects cut by a tile edge are stitched back together
OV_CACHE = Path(os.environ.get("OWLSEG_OV_CACHE", Path.home() / ".cache" / "owlseg-openvino"))
F32 = {"INFERENCE_PRECISION_HINT": "f32"}


@dataclass
class Candidate:
    """One SAM mask for one OWLv2 box, in image px."""
    box: np.ndarray       # x0, y0, x1, y1 (int) of the mask
    mask: np.ndarray      # bool, cropped to box: shape (y1 - y0, x1 - x0)
    score: float          # SAM mask quality x OWLv2 objectness
    cut: tuple = (False, False, False, False)  # mask touches an inner tile edge on the left/top/right/bottom


# ---- export wrappers: plain tensor in / tensors out ----------------------------------------------------
class _OwlObjectness(torch.nn.Module):
    """OWLv2 image half: pixels -> (objectness logits (1, 3600), boxes cxcywh in 0..1 (1, 3600, 4))."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        feature_map, _ = self.m.image_embedder(pixel_values=pixel_values)
        b, h, w, d = feature_map.shape
        feats = feature_map.reshape(b, h * w, d)
        return self.m.objectness_predictor(feats), self.m.box_predictor(feats, feature_map)


class _SamEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m.get_image_embeddings(pixel_values)


class _SamBoxDecoder(torch.nn.Module):
    """(image embedding, boxes (1, BOX_BATCH, 4) in SAM's 1024 frame) -> (iou (1, B, 1), low-res logits)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image_embeddings, input_boxes):
        out = self.m(image_embeddings=image_embeddings, input_boxes=input_boxes, multimask_output=False)
        return out.iou_scores, out.pred_masks


def _torch_runner(module: torch.nn.Module):
    """Give a PyTorch module the same call style as a compiled OpenVINO model: list of arrays in, list out."""
    module = module.eval()

    def run(inputs):
        with torch.inference_mode():
            out = module(*[torch.from_numpy(np.ascontiguousarray(x)) for x in inputs])
        return [o.numpy() for o in (out if isinstance(out, tuple) else (out,))]

    return run


def intel_gpu_available() -> bool:
    if os.environ.get("OWLSEG_DEVICE", "").lower() == "cpu":
        return False
    try:
        import openvino as ov
        return "GPU" in ov.Core().available_devices
    except Exception:  # openvino missing or no GPU driver
        return False


def _ov_dir() -> Path:
    import openvino as ov
    import transformers
    return OV_CACHE / f"ov{ov.__version__.split('-')[0]}_tf{transformers.__version__}"


def models_converted() -> bool:
    """True when the OpenVINO models exist already, i.e. loading takes seconds instead of a minute."""
    try:
        return all((_ov_dir() / f"{n}.xml").exists() for n in ("owl", "sam_encoder", "sam_box_decoder"))
    except Exception:
        return False


class OwlSam:
    def __init__(self, on_status=None):
        from transformers import Owlv2ImageProcessor, SamProcessor

        say = on_status or (lambda msg: None)
        self.owl_ip = Owlv2ImageProcessor.from_pretrained(OWL_MODEL)
        self.sam_proc = SamProcessor.from_pretrained(SAM_MODEL)
        self.sam_ip = self.sam_proc.image_processor
        self.backend = "torch"
        if os.environ.get("OWLSEG_BACKEND", "").lower() != "torch":
            try:
                self._load_openvino(say)
            except Exception as e:  # conversion / driver problem: fall back rather than fail
                print(f"[owlseg] OpenVINO unavailable ({e.__class__.__name__}: {e}); using PyTorch on the CPU.")
                self.backend = "torch"
        if self.backend == "torch":
            from transformers import Owlv2ForObjectDetection, SamModel
            say("Loading OWLv2 and SAM...")
            self.owl = _torch_runner(_OwlObjectness(Owlv2ForObjectDetection.from_pretrained(OWL_MODEL)))
            sam = SamModel.from_pretrained(SAM_MODEL).eval()
            self.sam_encoder, self.box_decoder = _torch_runner(_SamEncoder(sam)), _torch_runner(_SamBoxDecoder(sam))
            self.parallel = False
        self._pool = ThreadPoolExecutor(max_workers=1) if self.parallel else None

    def _load_openvino(self, say) -> None:
        import openvino as ov

        sam_device = "GPU" if intel_gpu_available() else "CPU"
        core = ov.Core()
        d = _ov_dir()
        d.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(d / "kernel_cache")})
        sam = owl = None
        for name in ("owl", "sam_encoder", "sam_box_decoder"):
            if (d / f"{name}.xml").exists():
                continue
            say("Preparing the models for this computer (first start only, about a minute)...")
            from transformers import Owlv2ForObjectDetection, SamModel
            if name == "owl":
                owl = owl or Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).eval()
                module, example = _OwlObjectness(owl), (torch.zeros(1, 3, OWL_SIZE, OWL_SIZE),)
            else:
                sam = sam or SamModel.from_pretrained(SAM_MODEL).eval()
                if name == "sam_encoder":
                    module, example = _SamEncoder(sam), (torch.zeros(1, 3, 1024, 1024),)
                else:
                    module, example = _SamBoxDecoder(sam), (torch.zeros(1, 256, 64, 64), torch.zeros(1, BOX_BATCH, 4))
            with torch.inference_mode():
                ov.save_model(ov.convert_model(module.eval(), example_input=example), d / f"{name}.xml",
                              compress_to_fp16=False)
        del sam, owl
        say("Loading the models onto " + ("the CPU and the Intel GPU..." if sam_device == "GPU" else "the CPU..."))
        self.owl = core.compile_model(d / "owl.xml", "CPU", F32)
        self.sam_encoder = core.compile_model(d / "sam_encoder.xml", sam_device, F32)
        self.box_decoder = core.compile_model(d / "sam_box_decoder.xml", sam_device, F32)
        self.backend = f"openvino (OWLv2 on CPU, SAM on {sam_device})"
        self.parallel = sam_device == "GPU"  # OWLv2 on the CPU overlaps SAM's encoder on the GPU

    # ---- steps -------------------------------------------------------------------------------------
    def _embed(self, image: Image.Image):
        base = self.sam_ip(images=image, return_tensors="pt")
        emb = self.sam_encoder([base["pixel_values"].numpy()])[0]
        return emb, base["original_sizes"], base["reshaped_input_sizes"]

    def _find_boxes(self, image: Image.Image, min_objectness: float, nms_iou: float):
        w, h = image.size
        px = self.owl_ip(images=image, return_tensors="np")["pixel_values"]
        r = self.owl([px])  # OpenVINO results support r[0], r[1] but not slicing
        logits, cxcywh = r[0], r[1]
        obj = torch.from_numpy(logits[0]).sigmoid()
        cx, cy, bw, bh = torch.from_numpy(cxcywh[0]).unbind(-1)
        side = max(w, h)  # OWLv2 pads the image to a square before resizing
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * side
        keep = obj >= min_objectness
        boxes, obj = boxes[keep], obj[keep]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        k = nms(boxes, obj, nms_iou)
        return boxes[k].numpy(), obj[k].numpy()

    def _outline(self, emb, boxes: np.ndarray, tile_hw: tuple[int, int], reshaped_hw: tuple[int, int], report):
        """SAM masks for box prompts (tile px). Returns [(box x0,y0,x1,y1 int, mask cropped to that box, iou)].

        Masks are upsampled only inside each object's box: a full-size mask per object would need gigabytes on a
        long belt recording.
        """
        h, w = tile_hw
        rh, rw = reshaped_hw
        scale = 1024 / max(h, w)  # SAM resizes the longest side to 1024
        prompts = (boxes * scale).astype(np.float32)
        out = []
        n_batches = max(1, -(-len(prompts) // BOX_BATCH))
        for k, i in enumerate(range(0, len(prompts), BOX_BATCH)):
            report(k / n_batches)
            chunk = prompts[i:i + BOX_BATCH]
            n = len(chunk)
            if n < BOX_BATCH:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], BOX_BATCH - n, 0)])
            r = self.box_decoder([emb, chunk[None]])
            iou, low = np.asarray(r[0])[0, :n, 0], np.asarray(r[1])[0, :n, 0]  # low-res logits (n, 256, 256)
            up = F.interpolate(torch.from_numpy(low)[:, None], size=(1024, 1024), mode="bilinear",
                               align_corners=False)[:, 0, :rh, :rw]  # (n, rh, rw) logits, padding removed
            for j in range(n):
                ys, xs = torch.nonzero(up[j] > 0, as_tuple=True)
                if len(ys) == 0:
                    out.append(None)
                    continue
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                Y0, Y1 = int(y0 / scale), min(h, int(np.ceil(y1 / scale)))
                X0, X1 = int(x0 / scale), min(w, int(np.ceil(x1 / scale)))
                crop = F.interpolate(up[j, y0:y1, x0:x1][None, None], size=(Y1 - Y0, X1 - X0), mode="bilinear",
                                     align_corners=False)[0, 0] > 0
                out.append((np.array([X0, Y0, X1, Y1]), crop.numpy(), float(iou[j])))
        report(1.0)
        return out

    def run(self, image: Image.Image, min_objectness: float = 0.1, nms_iou: float = 0.7, report=None,
            timings: dict | None = None, tile: int = TILE, overlap: int = TILE_OVERLAP) -> list[Candidate]:
        """Every object OWLv2 proposes, outlined by SAM, as Candidates in image px.

        Images larger than `tile` are processed in overlapping tiles so OWLv2 (960 px input) and SAM (1024 px)
        see objects near full resolution; a whole 2048 x 30000 belt recording squeezed into 960 px loses almost
        every object. score = SAM's predicted mask quality x OWLv2 objectness.
        report(stage, fraction within the tile, tile number, number of tiles) drives a progress bar.
        """
        import time

        report = report or (lambda stage, frac: None)
        t = timings if timings is not None else {}
        for k in (("owl+sam_embed",) if self.parallel else ("owl", "sam_embed")) + ("sam_outline",):
            t.setdefault(k, 0.0)
        w, h = image.size
        tiles = tile_grid(w, h, tile, overlap)
        out: list[Candidate] = []
        for n, (x0, y0, x1, y1) in enumerate(tiles):
            sub = image if len(tiles) == 1 else image.crop((x0, y0, x1, y1))
            step = lambda stage, frac, n=n: report(stage, frac, n + 1, len(tiles))  # noqa: E731
            t0 = time.perf_counter()
            step("find", 0.0)
            if self.parallel:  # SAM encodes the tile on the GPU while OWLv2 looks for objects on the CPU
                embed_job = self._pool.submit(self._embed, sub)
                boxes, obj = self._find_boxes(sub, min_objectness, nms_iou)
                emb, _, reshaped = embed_job.result()
                t["owl+sam_embed"] += time.perf_counter() - t0
            else:
                boxes, obj = self._find_boxes(sub, min_objectness, nms_iou)
                t["owl"] += time.perf_counter() - t0
                step("embed", 0.0)
                t0 = time.perf_counter()
                emb, _, reshaped = self._embed(sub)
                t["sam_embed"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            rh, rw = (int(v) for v in reshaped[0])
            masks = self._outline(emb, boxes, (y1 - y0, x1 - x0), (rh, rw), lambda f: step("outline", f))
            t["sam_outline"] += time.perf_counter() - t0
            for found, ob in zip(masks, obj):
                if found is None:
                    continue
                b, m, iou = found
                # which sides of the mask touch a tile edge that lies inside the image (object may be cut there)
                cut = (x0 > 0 and b[0] <= 1, y0 > 0 and b[1] <= 1,
                       x1 < w and b[2] >= x1 - x0 - 1, y1 < h and b[3] >= y1 - y0 - 1)
                out.append(Candidate(b + np.array([x0, y0, x0, y0]), m, iou * float(ob), cut))
        return out


def tile_grid(w: int, h: int, tile: int = TILE, overlap: int = TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles (x0, y0, x1, y1) covering the image; one tile when it is no bigger than `tile`."""
    def starts(n):
        if n <= tile:
            return [0]
        step = tile - overlap
        return list(range(0, n - tile, step)) + [n - tile]
    tw, th = min(w, tile), min(h, tile)
    return [(x, y, x + tw, y + th) for y in starts(h) for x in starts(w)]
