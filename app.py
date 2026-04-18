"""
Gradio UI for the Neuraxon Image Evolution Engine.

Flow:
    1. Upload an image
    2. Seed a random population of filter-neurons
    3. Each neuron renders a variant of the image
    4. Check the variants you like -> click Evolve -> unpicked neurons
       lose health and die, picked ones are cloned and mutated.
"""

from __future__ import annotations

import os
import tempfile

import gradio as gr
from PIL import Image

from image_evolution import ImageEvolutionEngine


def _format_stats(engine: ImageEvolutionEngine) -> str:
    s = engine.stats()
    hist = s["filter_histogram"]
    top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
    hist_md = "\n".join(f"- `{k}`: {v}" for k, v in top) or "- (empty)"
    last = s["last_gen_stats"]
    return (
        f"**Generation:** {s['generation']}  \n"
        f"**Population:** {s['population']}  \n"
        f"**Last gen:** births={last.get('births', 0)}, "
        f"deaths={last.get('deaths', 0)}, "
        f"survivors={last.get('survivors', 0)}, "
        f"picked={last.get('picked', 0)}\n\n"
        f"**Filter-op population (top 8):**\n{hist_md}"
    )


def _gallery_and_choices(engine: ImageEvolutionEngine):
    previews = engine.previews()
    labels = engine.labels()
    ids = engine.neuron_ids()
    gallery = [(img, f"#{nid}  {lbl}") for img, lbl, nid in zip(previews, labels, ids)]
    choice_labels = [f"#{nid}" for nid in ids]
    return gallery, gr.CheckboxGroup(choices=choice_labels, value=[])


def _parse_selected(checkbox_values: list[str]) -> list[int]:
    out = []
    for v in checkbox_values or []:
        s = v.strip().lstrip("#")
        try:
            out.append(int(s))
        except ValueError:
            continue
    return out


def build_ui() -> gr.Blocks:
    engine = ImageEvolutionEngine()

    with gr.Blocks(title="Neuraxon Image Evolution") as demo:
        gr.Markdown(
            "# Neuraxon Image Evolution Engine\n"
            "Upload an image, seed a population of filter-neurons, then pick your "
            "favorites each generation. Unpicked neurons lose health and die; "
            "picked ones are cloned and mutated. Each neuron is a *chain* of ops, "
            "so a single neuron can apply the same filter multiple times with "
            "different parameters."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Source image", height=280)
                pop_size = gr.Slider(4, 24, value=12, step=1, label="Population size")
                seed_btn = gr.Button("Seed population", variant="primary")
                evolve_btn = gr.Button("Evolve next generation", variant="primary")
                save_btn = gr.Button("Save best (by age)")
                stats_md = gr.Markdown("_Population not seeded yet._")
                saved_path = gr.Textbox(label="Saved to", interactive=False)
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Population", columns=4, height=640, object_fit="contain")
                picks = gr.CheckboxGroup(choices=[], label="Mark as favorites (survivors/parents)")

        def on_seed(pil_img, size):
            if pil_img is None:
                return [], gr.CheckboxGroup(choices=[], value=[]), "_Upload an image first._", ""
            engine.load_image(pil_img)
            engine.seed(int(size))
            g, cg = _gallery_and_choices(engine)
            return g, cg, _format_stats(engine), ""

        def on_evolve(selected):
            if engine.source is None:
                return [], gr.CheckboxGroup(choices=[], value=[]), "_Seed the population first._", ""
            ids = _parse_selected(selected)
            engine.step(ids)
            g, cg = _gallery_and_choices(engine)
            return g, cg, _format_stats(engine), ""

        def on_save():
            if engine.source is None:
                return "_Nothing to save — seed first._"
            fd, path = tempfile.mkstemp(prefix="neuraxon_best_", suffix=".png")
            os.close(fd)
            saved = engine.save_best(path)
            return saved or "_No best neuron yet._"

        seed_btn.click(on_seed, inputs=[image_in, pop_size], outputs=[gallery, picks, stats_md, saved_path])
        evolve_btn.click(on_evolve, inputs=[picks], outputs=[gallery, picks, stats_md, saved_path])
        save_btn.click(on_save, outputs=[saved_path])

    return demo


if __name__ == "__main__":
    build_ui().launch()
