"""Gradio UI: upload an image -> every object highlighted in light blue; tick objects in the list to show/hide."""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from owlseg.pipeline import OwlSamPipeline, Settings
from owlseg.visualize import object_label, viewer_html

PORT = 7862
PIPE = OwlSamPipeline()

CSS = """
.gradio-container{max-width:1600px !important}
#obj-list .wrap{display:flex;flex-direction:column;flex-wrap:nowrap;gap:2px;max-height:560px;overflow-y:auto}
#obj-list label{width:100%;font-variant-numeric:tabular-nums}
#obj-total{font-size:20px;font-weight:700}
#ov-wrap{display:flex;justify-content:center;align-items:center;min-height:640px}
#ov-wrap .ov-box{position:relative;display:inline-block;line-height:0}
#ov-wrap .ov-base{display:block;max-width:100%;max-height:640px}
#ov-wrap .ov{position:absolute;pointer-events:none}
"""
EMPTY_VIEW = '<div id="ov-wrap" style="opacity:.6">The highlighted objects appear here.</div>'

# Ticking runs only in the browser: each object's highlight is a ready-made layer that is shown or hidden.
# The scripts read the tick boxes on the page (Gradio hands js handlers the value from before the click).
# List row i and layer i are the same object.
JS_SELECT = """() => {
  const boxes = [...document.querySelectorAll('#obj-list input[type=checkbox]')];
  const layers = document.querySelectorAll('#ov-wrap .ov');
  boxes.forEach((b, i) => { if (layers[i]) layers[i].style.display = b.checked ? '' : 'none'; });
  const all = document.querySelector('#mask-all input');
  if (all) all.checked = boxes.length > 0 && boxes.every(b => b.checked);
}"""
JS_MASK_ALL = """() => {
  const on = document.querySelector('#mask-all input').checked;
  // click the boxes that differ, so Gradio's list value follows; each click re-runs JS_SELECT
  document.querySelectorAll('#obj-list input[type=checkbox]').forEach(b => { if (b.checked !== on) b.click(); });
  document.querySelectorAll('#ov-wrap .ov').forEach(e => { e.style.display = on ? '' : 'none'; });
}"""


def _total(n: int) -> str:
    return f"Total objects detected: {n}" if n else "No objects found. Try a lower objectness threshold."


def run(image_path, objectness, min_area, max_area, label_style, progress=gr.Progress()):
    if not image_path:
        raise gr.Error("Please upload an image first.")
    progress(0.0, desc="Starting...")
    s = Settings(min_objectness=objectness, min_area=min_area / 100, max_area=max_area / 100,
                 label_style=label_style)
    res = PIPE.run(Image.open(image_path), s, progress=lambda frac, desc: progress(frac, desc=desc))
    progress(1.0, desc="Saving results...")
    png, js = res.save(Path(image_path).stem)
    info = "**Timings (s):** " + " · ".join(f"{k} {v}" for k, v in res.timings.items()) + f"\n\nSaved to `{png}`"
    area = res.image.width * res.image.height
    ids = [o["id"] for o in res.objects]
    choices = [(object_label(o, area), o["id"]) for o in res.objects]
    state = {"image": res.image, "objects": res.objects, "masks": res.masks}
    view = viewer_html(res.image, res.objects, res.masks, label_style=label_style)
    return (view, _total(len(ids)), gr.update(value=bool(ids), interactive=bool(ids)),
            gr.update(choices=choices, value=ids), info, [str(png), str(js)], state)


def relabel(state, label_style):
    """Label style changed: rebuild the layers (the only display change that needs Python); JS_SELECT then
    re-applies the ticks."""
    if not state:
        return EMPTY_VIEW
    return viewer_html(state["image"], state["objects"], state["masks"], label_style=label_style)


with gr.Blocks(title="OWLv2 + SAM Object Finder", css=CSS) as demo:
    gr.Markdown(
        "# OWLv2 + SAM Object Finder\n"
        "OWLv2 finds every object, SAM outlines it. No prompt, no labels: objects are numbered and counted. "
        "Tick objects in the list to highlight them."
    )
    state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            inp = gr.Image(type="filepath", label="Input image", height=300)
            btn = gr.Button("Find objects", variant="primary")
            with gr.Accordion("Settings", open=False):
                objectness = gr.Slider(0.02, 0.5, 0.1, step=0.01, label="Objectness threshold",
                                       info="How sure OWLv2 must be that a box holds a whole object. "
                                            "Lower finds more, but also more junk")
                min_area = gr.Slider(0.05, 5, 0.2, step=0.05, label="Min object size (% of image)")
                max_area = gr.Slider(1, 80, 25, step=1, label="Max object size (% of image)",
                                     info="Bigger regions are treated as background")
                label_style = gr.Radio(["auto", "full", "number", "none"], value="auto", label="Object labels",
                                       info="auto = '#id score' up to 30 objects, then numbers only")
        with gr.Column(scale=3):
            view = gr.HTML(EMPTY_VIEW, label="Objects")
        with gr.Column(scale=1, min_width=260):
            total = gr.Markdown("Upload an image and press **Find objects**.", elem_id="obj-total")
            mask_all = gr.Checkbox(False, label="Mask all", interactive=False, elem_id="mask-all")
            # interactive=True: no Python event reads these boxes, so Gradio would otherwise disable them
            obj_list = gr.CheckboxGroup([], label="Objects (size · score)", elem_id="obj-list", interactive=True)
    info = gr.Markdown()
    out_files = gr.File(label="Results: annotated image (all objects) and JSON", file_count="multiple")

    btn.click(run, [inp, objectness, min_area, max_area, label_style],
              [view, total, mask_all, obj_list, info, out_files, state])
    # fn=None + js: these run only in the browser. .input fires on user clicks only, so the boxes don't loop.
    mask_all.input(None, None, None, js=JS_MASK_ALL)
    obj_list.input(None, None, None, js=JS_SELECT)
    label_style.input(relabel, [state, label_style], view).then(None, None, None, js=JS_SELECT)


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=PORT, inbrowser=not os.environ.get("NO_BROWSER"))
