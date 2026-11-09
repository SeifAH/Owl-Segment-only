# How it works

## 1. Why not a grid of points

`sam-segment-only` asks SAM for a mask at each of 24 × 24 = 576 evenly spaced points. SAM answers "what region is under this point?". On a sorting belt, touching white or clear items often come back as **one region**, and small items between grid points are missed. The grid also wastes most of its points on the belt.

This app asks a detector first **where the objects are**, then asks SAM to outline each one. A box prompt tells SAM exactly which object is meant, so touching items stay separate.

## 2. OWLv2 without class names

OWLv2 (`google/owlv2-base-patch16-ensemble`) is an open-vocabulary detector: normally you give it class names. It also has an **objectness head**: for each of its 60 × 60 = 3600 image patches it predicts one box and a score for "there is a whole object here", without any class name. Only the image half of the model runs here (image encoder, objectness head, box head); the text half is not loaded.

Steps:
1. The image is padded to a square and resized to 960 × 960.
2. OWLv2 returns 3600 boxes with objectness scores.
3. Boxes with objectness ≥ 0.1 are kept; non-maximum suppression (IoU 0.7) removes near-duplicates.

## 3. SAM: one mask per box

SAM ViT-B encodes the image once (1024 × 1024 → 64 × 64 × 256 embedding). The boxes are then decoded in batches of 32, one mask per box (`multimask_output=False`). Masks are upsampled to full resolution on the CPU.

Score of an object = SAM's predicted mask quality × OWLv2 objectness.

## 4. Filters (`filters.py`)

Masks are visited best score first. A mask is dropped when it is:
- smaller than 0.2% or larger than 25% of the image,
- wider or taller than 85% of the image (belt, floor),
- more than 50% on top of objects already kept (a second box on the same object, or a part of it).

Objects are numbered in reading order (row bands top to bottom, then left to right).

## 5. Where each model runs

| | OWLv2 | SAM encoder | SAM box decoder |
|---|---|---|---|
| Laptop with Intel GPU (default) | OpenVINO, CPU | OpenVINO, Intel GPU | OpenVINO, Intel GPU |
| No Intel GPU | OpenVINO, CPU | OpenVINO, CPU | OpenVINO, CPU |
| OpenVINO fails | PyTorch, CPU | PyTorch, CPU | PyTorch, CPU |

With the Intel GPU, OWLv2 (CPU) and the SAM encoder (GPU) run **at the same time**. That is why the first step's time is shown as `owl+sam_embed`.

All models run in **fp32**. Choices tested on 13 frames:

| Variant | Found (IoU ≥ 0.5) | Time per frame |
|---|---|---|
| OWLv2 on GPU, then SAM on GPU (fp32) | 81% | 15.4 s |
| OWLv2 on GPU in fp16 (GPU default) | 78% | 7.5 s |
| OWLv2 on the NPU | could not compile (dynamic shapes) | – |
| **OWLv2 on CPU ∥ SAM on GPU, fp32 (used)** | **81%** | **8.4–8.7 s** |

Overrides: `OWLSEG_DEVICE=cpu` (skip the GPU), `OWLSEG_BACKEND=torch` (skip OpenVINO).

The first start converts the three models to OpenVINO format and saves them in `%USERPROFILE%\.cache\owlseg-openvino` (about a minute); compiled GPU kernels are cached there too.

## 6. Measured results

13 frames (1, 5, …, 49) of `before-video-002`, scored against the ground-truth masks. Each white region of a mask is one target item (149 in total). An item counts as found when one predicted object overlaps it with IoU ≥ 0.5.

| Method | Found @0.5 | Found @0.75 | Mean IoU | Objects/frame | s/frame |
|---|---|---|---|---|---|
| SAM grid (`sam-segment-only`) | 66% | 51% | 0.64 | 37.2 | 14.3 |
| Florence-2 boxes → SAM | 43% | 35% | 0.42 | 20.1 | 10.1 |
| OWLv2 + grid in the gaps | 82% | 60% | 0.73 | 43.0 | 31.1 |
| **OWLv2 → SAM (this app)** | **81%** | **60%** | **0.72** | **39.3** | **8.7** |

Per frame time in this app: OWLv2 ∥ SAM encoder 4.4–6.4 s, SAM outlines 1.6–2.8 s, filter + drawing 0.5–0.8 s.

Limits of the test:
- One video only.
- The masks mark only some items (mostly white and clear ones), so extra or false objects are not measured.
