"""Gradio UI: upload an image -> every object outlined and numbered (no labels) + scrollable list."""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image

from owlseg.pipeline import OwlSamPipeline, Settings

PORT = 7862
PIPE = OwlSamPipeline()
EMPTY_LIST = '<div style="padding:12px;opacity:.6">Upload an image and press <b>Find objects</b>.</div>'


def run(image_path, objectness, min_area, max_area, show_masks, label_style, progress=gr.Progress()):
    if not image_path:
        raise gr.Error("Please upload an image first.")
    progress(0.0, desc="Starting...")
    s = Settings(min_objectness=objectness, min_area=min_area / 100, max_area=max_area / 100,
                 show_masks=show_masks, label_style=label_style)
    res = PIPE.run(Image.open(image_path), s, progress=lambda frac, desc: progress(frac, desc=desc))
    progress(1.0, desc="Saving results...")
    png, js = res.save(Path(image_path).stem)
    info = "**Timings (s):** " + " · ".join(f"{k} {v}" for k, v in res.timings.items()) + f"\n\nSaved to `{png}`"
    return res.annotated, res.list_html, info, str(js)


with gr.Blocks(title="OWLv2 + SAM Object Finder", css=".gradio-container{max-width:1600px !important}") as demo:
    gr.Markdown(
        "# OWLv2 + SAM Object Finder\n"
        "OWLv2 finds every object, SAM outlines it. No prompt, no labels: objects are numbered and counted."
    )
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
                show_masks = gr.Checkbox(True, label="Show masks")
                label_style = gr.Radio(["auto", "full", "number", "none"], value="auto", label="Box labels",
                                       info="auto = '#id score' up to 30 objects, then numbers only")
        with gr.Column(scale=3):
            out_img = gr.Image(type="pil", label="Objects", height=640, show_download_button=True)
        with gr.Column(scale=1, min_width=260):
            out_list = gr.HTML(EMPTY_LIST)
    info = gr.Markdown()
    out_json = gr.File(label="Result JSON")

    btn.click(run, [inp, objectness, min_area, max_area, show_masks, label_style],
              [out_img, out_list, info, out_json])


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=PORT, inbrowser=not os.environ.get("NO_BROWSER"))
