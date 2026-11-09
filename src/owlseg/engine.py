"""OWLv2 finds the objects (boxes), SAM outlines each one (masks). No class names, no labels.

OWLv2 is an open-vocabulary detector; here only its image half runs. For each of its 3600 image patches
(60 x 60 at 960 px) it predicts a box and an *objectness* score: "how likely is there a whole object here",
independent of any class name. Boxes above `min_objectness` survive non-maximum suppression and are passed to
SAM as box prompts; SAM returns one mask per box.

Where each model runs (chosen automatically, fastest first):
  * NVIDIA GPU (CUDA): everything on the GPU via PyTorch in strict fp32 (TF32 off), including the OWLv2 and SAM
    preprocessing, mask upsampling and the object filters; only the kept objects come back to the CPU.
  * OWLv2 on the CPU and SAM on the Intel integrated GPU, both via OpenVINO in fp32, run at the same time
    (~8 s per image on a Core Ultra 7 155U).
  * No Intel GPU: everything via OpenVINO on the CPU, one after the other.
  * OpenVINO fails: PyTorch on the CPU (slowest, same results).
Set OWLSEG_DEVICE=cpu to skip any GPU, OWLSEG_BACKEND=torch to force PyTorch on the CPU, OWLSEG_BACKEND=openvino
to skip CUDA.

fp32 everywhere is deliberate: the GPU's default fp16 mode found fewer objects in tests (78% vs 81%).
Converted models are saved once to OV_CACHE (first start ~1 minute); compiled GPU kernels are cached there too.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import nms

from .models import OWL_MODEL, SAM_MODEL

BOX_BATCH = 32  # SAM box prompts per decoder call (static shape for OpenVINO; last batch is padded)
CUDA_BOX_BUCKETS = (16, 32, 64, 128)  # SAM box prompts per decoder call on CUDA, padded up to one of these sizes
CUDA_BOX_BATCH = CUDA_BOX_BUCKETS[-1]  # (fixed shapes: each size is compiled once)
CUDA_UPSAMPLE_BATCH = 32  # masks upsampled to full resolution at once on CUDA (bounds GPU memory)
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


def cuda_available() -> bool:
    if os.environ.get("OWLSEG_DEVICE", "").lower() == "cpu":
        return False
    if os.environ.get("OWLSEG_BACKEND", "").lower() in ("torch", "openvino"):
        return False
    return torch.cuda.is_available()


def intel_gpu_available() -> bool:
    if os.environ.get("OWLSEG_DEVICE", "").lower() == "cpu":
        return False
    try:
        import openvino as ov
        core = ov.Core()
        # OpenVINO also lists NVIDIA cards (via OpenCL) as "GPU", but cannot compile these models for them
        return any(d.startswith("GPU") and "intel" in core.get_property(d, "FULL_DEVICE_NAME").lower()
                   for d in core.available_devices)
    except Exception:  # openvino missing or no GPU driver
        return False


# ---- exact GPU ports of the Hugging Face preprocessing (results match to float32 rounding) ---------------
def _mirror(idx: torch.Tensor, n: int) -> torch.Tensor:
    """scipy.ndimage 'mirror' boundary: reflect about the first and last pixel centres."""
    if n == 1:
        return torch.zeros_like(idx)
    period = 2 * (n - 1)
    idx = idx.remainder(period)
    return torch.where(idx > n - 1, period - idx, idx)


def _gauss_1d(x: torch.Tensor, sigma: float, dim: int) -> torch.Tensor:
    """scipy.ndimage.gaussian_filter1d (truncate=4, mode='mirror'), computed in float64 like scipy."""
    radius = int(4.0 * sigma + 0.5)
    k = torch.arange(-radius, radius + 1, dtype=torch.float64)
    k = torch.exp(-0.5 / (sigma * sigma) * k ** 2)
    k = (k / k.sum()).tolist()
    n = x.shape[dim]
    pos = torch.arange(n, device=x.device)
    xd = x.double()
    out = torch.zeros_like(xd)
    for j, wj in enumerate(k):
        out += wj * xd.index_select(dim, _mirror(pos + (j - radius), n))
    return out.float()


def _zoom_linear(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    """One axis of scipy.ndimage.zoom(order=1, mode='mirror', grid_mode=True), in float64."""
    n = x.shape[dim]
    c = (torch.arange(size, dtype=torch.float64, device=x.device) + 0.5) * (n / size) - 0.5
    c = torch.where(c < 0, -c, c)
    c = torch.where(c > n - 1, 2 * (n - 1) - c, c)
    i0 = c.floor()
    w1 = c - i0
    i0 = i0.long()
    shape = [1] * x.dim()
    shape[dim] = size
    w1 = w1.view(shape)
    return x.index_select(dim, i0) * (1 - w1) + x.index_select(dim, _mirror(i0 + 1, n)) * w1


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
        self.device = "cpu"
        if cuda_available():
            try:
                self._load_cuda(say)
            except Exception as e:  # driver / out-of-memory problem: fall back rather than fail
                print(f"[owlseg] CUDA unavailable ({e.__class__.__name__}: {e}); trying OpenVINO.")
                self.device = "cpu"
        if self.device == "cpu" and os.environ.get("OWLSEG_BACKEND", "").lower() != "torch":
            try:
                self._load_openvino(say)
            except Exception as e:  # conversion / driver problem: fall back rather than fail
                print(f"[owlseg] OpenVINO unavailable ({e.__class__.__name__}: {e}); using PyTorch on the CPU.")
                self.backend = "torch"
        if self.device == "cpu" and self.backend == "torch":
            from transformers import Owlv2ForObjectDetection, SamModel
            say("Loading OWLv2 and SAM...")
            self.owl = _torch_runner(_OwlObjectness(Owlv2ForObjectDetection.from_pretrained(OWL_MODEL)))
            sam = SamModel.from_pretrained(SAM_MODEL).eval()
            self.sam_encoder, self.box_decoder = _torch_runner(_SamEncoder(sam)), _torch_runner(_SamBoxDecoder(sam))
            self.parallel = False
        self._pool = ThreadPoolExecutor(max_workers=1) if self.parallel else None

    def _load_cuda(self, say) -> None:
        from transformers import Owlv2ForObjectDetection, SamModel

        # strict fp32: TF32 (on by default for cuDNN convolutions) would change the numbers
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dev = torch.device("cuda")
        say("Loading OWLv2 and SAM onto the NVIDIA GPU...")
        owl = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL)
        self.owl_cuda = _OwlObjectness(owl).eval().to(dev)
        del owl.owlv2.text_model, owl.owlv2.text_projection  # only the image half is used
        self.sam_cuda = SamModel.from_pretrained(SAM_MODEL).eval().to(dev)
        if os.environ.get("OWLSEG_EAGER_ATTENTION") != "1":
            from .fast_attention import use_fused_attention
            use_fused_attention(owl.owlv2.vision_model, self.sam_cuda.vision_encoder)
        f32 = dict(dtype=torch.float32, device=dev)
        self.owl_mean = torch.tensor(self.owl_ip.image_mean, **f32).view(3, 1, 1)
        self.owl_std = torch.tensor(self.owl_ip.image_std, **f32).view(3, 1, 1)
        self.sam_mean = torch.tensor(self.sam_ip.image_mean, **f32).view(3, 1, 1)
        self.sam_std = torch.tensor(self.sam_ip.image_std, **f32).view(3, 1, 1)
        self.device = "cuda"
        self.parallel = True  # the CPU prepares SAM's input while the GPU runs OWLv2
        self.backend = f"cuda (OWLv2 + SAM on {torch.cuda.get_device_name(dev)}, fp32"
        warm = Image.new("RGB", (64, 48), (128, 128, 128))
        if os.environ.get("OWLSEG_COMPILE", "1") != "0":
            # fused fp32 kernels, fixed input shapes so nothing recompiles later; torch caches them on disk
            say("Compiling the models for this GPU (first start a few minutes, later about a minute)...")
            dec = self.sam_cuda.mask_decoder
            eager = self.owl_cuda, self.sam_cuda.vision_encoder, dec.transformer
            try:
                self.owl_cuda = torch.compile(self.owl_cuda, dynamic=False)
                self.sam_cuda.vision_encoder = torch.compile(self.sam_cuda.vision_encoder, dynamic=False)
                dec.transformer = torch.compile(dec.transformer, dynamic=False)  # the decoder's costly part
                self._warm_up(warm)
                self.backend += ", compiled"
            except Exception as e:  # no C compiler / triton problem: the eager models give the same results
                print(f"[owlseg] torch.compile unavailable ({e.__class__.__name__}: {e}); running uncompiled.")
                self.owl_cuda, self.sam_cuda.vision_encoder, dec.transformer = eager
        self.backend += ")"
        say("Warming up the GPU...")
        self._warm_up(warm)

    def _warm_up(self, image: Image.Image) -> None:
        """One pass through every model and every box-batch size, so the first real image is not slower."""
        self._run_cuda(image, 1.0, 0.7, lambda *a: None, {})
        with torch.inference_mode():
            emb = self.sam_cuda.get_image_embeddings(self._sam_pixels_cuda(image)[0])
            for size in CUDA_BOX_BUCKETS:
                box = torch.tensor([[[0.0, 0.0, 32.0, 32.0]]], device=emb.device).expand(1, size, 4)
                self.sam_cuda(image_embeddings=emb, input_boxes=box, multimask_output=False)

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

    # ---- CUDA steps: same maths as the steps above, all on the GPU -------------------------------------
    def _owl_pixels_cuda(self, img: torch.Tensor) -> torch.Tensor:
        """Owlv2ImageProcessor on the GPU: rescale, pad to a square with 0.5, anti-alias, resize, normalise."""
        h, w = img.shape[1:]
        side = max(h, w)
        x = (img.double() * self.owl_ip.rescale_factor).float()
        x = F.pad(x, (0, side - w, 0, side - h), value=0.5)
        lo, hi = x.min(), x.max()
        sigma = max(0.0, (side / OWL_SIZE - 1) / 2)
        if sigma > 1e-15:
            x = _gauss_1d(_gauss_1d(x, sigma, 1), sigma, 2)
        x = _zoom_linear(_zoom_linear(x.double(), OWL_SIZE, 1), OWL_SIZE, 2).float()
        x = x.clamp(lo, hi)
        return ((x - self.owl_mean) / self.owl_std)[None].contiguous()

    def _sam_pixels_cuda(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
        """SamImageProcessor: PIL resize on the CPU (exactly as Hugging Face), the rest on the GPU."""
        small = self.sam_ip.resize(np.asarray(image), size=self.sam_ip.size, resample=self.sam_ip.resample,
                                   input_data_format="channels_last")
        rh, rw = small.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(small)).cuda(non_blocking=True).permute(2, 0, 1)
        x = ((x.double() * self.sam_ip.rescale_factor).float() - self.sam_mean) / self.sam_std
        pad = self.sam_ip.pad_size
        return F.pad(x, (0, pad["width"] - rw, 0, pad["height"] - rh))[None].contiguous(), (rh, rw)

    @torch.inference_mode()
    def _run_cuda(self, image: Image.Image, min_objectness: float, nms_iou: float, report, t: dict):
        import time

        w, h = image.size
        t0 = time.perf_counter()
        report("find", 0.0)
        img = torch.from_numpy(np.array(image)).cuda().permute(2, 0, 1)
        logits, cxcywh = self.owl_cuda(self._owl_pixels_cuda(img))  # queued on the GPU ...
        sam_px, (rh, rw) = self._sam_pixels_cuda(image)  # ... while the CPU resizes for SAM
        emb = self.sam_cuda.get_image_embeddings(sam_px)
        obj = logits[0].sigmoid()
        cx, cy, bw, bh = cxcywh[0].unbind(-1)
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * max(w, h)
        keep = obj >= min_objectness
        boxes, obj = boxes[keep], obj[keep]
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        k = nms(boxes, obj, nms_iou)
        boxes, obj = boxes[k], obj[k]
        torch.cuda.synchronize()
        t["owl+sam_embed"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n = len(boxes)
        # SamProcessor._normalize_coordinates, in float64 like the original
        prompts = (boxes.double() * torch.tensor([rw / w, rh / h] * 2, dtype=torch.float64, device=boxes.device))
        prompts = prompts.float()
        masks = torch.empty((n, h, w), dtype=torch.bool, device=boxes.device)
        ious = torch.empty(n, dtype=torch.float32, device=boxes.device)
        n_batches = max(1, -(-n // CUDA_BOX_BATCH))
        for b, i in enumerate(range(0, n, CUDA_BOX_BATCH)):
            report("outline", b / n_batches)
            chunk = prompts[i:i + CUDA_BOX_BATCH]
            k = len(chunk)
            size = next(c for c in CUDA_BOX_BUCKETS if c >= k)
            chunk = torch.cat([chunk, chunk[-1:].expand(size - k, 4)])  # boxes are decoded independently
            out = self.sam_cuda(image_embeddings=emb, input_boxes=chunk[None], multimask_output=False)
            ious[i:i + k] = out.iou_scores[0, :k, 0]
            low = out.pred_masks[0, :k]  # (k, 1, 256, 256) low-res logits
            for j in range(0, len(low), CUDA_UPSAMPLE_BATCH):  # SamImageProcessor.post_process_masks
                m = F.interpolate(low[j:j + CUDA_UPSAMPLE_BATCH], (self.sam_ip.pad_size["height"],
                                  self.sam_ip.pad_size["width"]), mode="bilinear", align_corners=False)
                m = F.interpolate(m[..., :rh, :rw], (h, w), mode="bilinear", align_corners=False)
                masks[i + j:i + j + len(m)] = m[:, 0] > 0
        report("outline", 1.0)
        scores = (ious * obj).cpu().numpy()
        t["sam_outline"] = time.perf_counter() - t0
        return masks, scores, obj.cpu().numpy()

    def run(self, image: Image.Image, min_objectness: float = 0.1, nms_iou: float = 0.7, report=None,
            timings: dict | None = None) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Return (masks (N, H, W) bool tensor, scores (N,), objectness (N,)) for every box OWLv2 proposes.

        The masks stay on the GPU with CUDA. score = SAM's predicted mask quality x OWLv2 objectness.
        report(stage, fraction) drives a progress bar.
        """
        import time

        report = report or (lambda stage, frac: None)
        t = timings if timings is not None else {}
        if self.device == "cuda":
            return self._run_cuda(image, min_objectness, nms_iou, report, t)
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
            return torch.zeros((0, h, w), dtype=torch.bool), np.zeros(0, np.float32), np.zeros(0, np.float32)
        return torch.from_numpy(np.concatenate(masks)), np.concatenate(ious) * obj, obj
