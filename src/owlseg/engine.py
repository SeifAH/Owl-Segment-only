"""OWLv2 finds the objects (boxes), SAM outlines each one (masks). No class names, no labels.

OWLv2 is an open-vocabulary detector; here only its image half runs. For each of its 3600 image patches
(60 x 60 at 960 px) it predicts a box and an *objectness* score: "how likely is there a whole object here",
independent of any class name. Boxes above `min_objectness` survive non-maximum suppression and are passed to
SAM as box prompts; SAM returns one mask per box.

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

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from .models import OWL_MODEL, SAM_MODEL

BOX_BATCH = 32  # SAM box prompts per decoder call (static shape for OpenVINO; last batch is padded)
OWL_SIZE = 960  # OWLv2-base input resolution
OV_CACHE = Path(os.environ.get("OWLSEG_OV_CACHE", Path.home() / ".cache" / "owlseg-openvino"))
F32 = {"INFERENCE_PRECISION_HINT": "f32"}


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

    def run(self, image: Image.Image, min_objectness: float = 0.1, nms_iou: float = 0.7, report=None,
            timings: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (masks (N, H, W) bool, scores (N,), objectness (N,)) for every box OWLv2 proposes.

        score = SAM's predicted mask quality x OWLv2 objectness. report(stage, fraction) drives a progress bar.
        """
        import time

        report = report or (lambda stage, frac: None)
        t = timings if timings is not None else {}
        w, h = image.size
        t0 = time.perf_counter()
        report("find", 0.0)
        if self.parallel:  # SAM encodes the image on the GPU while OWLv2 looks for objects on the CPU
            embed_job = self._pool.submit(self._embed, image)
            boxes, obj = self._find_boxes(image, min_objectness, nms_iou)
            emb, orig, reshaped = embed_job.result()
            t["owl+sam_embed"] = time.perf_counter() - t0
        else:
            boxes, obj = self._find_boxes(image, min_objectness, nms_iou)
            t["owl"] = time.perf_counter() - t0
            report("embed", 0.0)
            t0 = time.perf_counter()
            emb, orig, reshaped = self._embed(image)
            t["sam_embed"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        prompts = self.sam_proc._normalize_coordinates(1024, boxes.astype(np.float32), (h, w), is_bounding_box=True)
        prompts = prompts.reshape(-1, 4).astype(np.float32)
        masks, ious = [], []
        n_batches = max(1, -(-len(prompts) // BOX_BATCH))
        for k, i in enumerate(range(0, len(prompts), BOX_BATCH)):
            report("outline", k / n_batches)
            chunk = prompts[i:i + BOX_BATCH]
            n = len(chunk)
            if n < BOX_BATCH:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], BOX_BATCH - n, 0)])
            r = self.box_decoder([emb, chunk[None]])
            iou, low = r[0], r[1]
            ious.append(iou[0, :n, 0])
            # (1, n, 1, 256, 256) low-res logits -> (n, 1, H, W) full-res logits
            full = self.sam_ip.post_process_masks(torch.from_numpy(low[:, :n]), orig, reshaped, binarize=False)[0]
            masks.append((full[:, 0] > 0).numpy())
        report("outline", 1.0)
        t["sam_outline"] = time.perf_counter() - t0
        if not masks:
            return np.zeros((0, h, w), bool), np.zeros(0, np.float32), np.zeros(0, np.float32)
        return np.concatenate(masks), np.concatenate(ious) * obj, obj
