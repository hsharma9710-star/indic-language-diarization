"""Web app: upload speech -> RTTM, language-coloured waveform, and pooled per-language audio.

Run locally:      python app.py            (then open http://127.0.0.1:7860)
Free hosting:     deploy this folder as a Hugging Face Space (SDK: Gradio, free CPU) - see README, "Website".
"""
import os
import tempfile

import gradio as gr

from ildiar.classes import LANG_NAMES, LANGS
from ildiar.engine import Diarizer
from ildiar.outputs import plot_diarization, split_by_language, write_rttm
from ildiar.weights import find_weights

MAX_SECONDS = float(os.environ.get("ILDIAR_MAX_SECONDS", 600))       # protects a free CPU host
_diarizer = None


def get_model():
    global _diarizer
    if _diarizer is None:
        _diarizer = Diarizer(find_weights(), device="cpu" if os.environ.get("ILDIAR_CPU") else None)
    return _diarizer


def run(audio_path, lang1, lang2, win_start, win_len, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("Please upload or record an audio file first.")
    langs = [l for l in (lang1, lang2) if l and l != "auto"]
    if len(set(langs)) != len(langs):
        raise gr.Error("Choose two different languages (or leave them on 'auto-detect').")
    d = get_model()
    from ildiar.audio import load_audio
    import numpy as np
    wav, sr = load_audio(audio_path)
    if len(wav) / sr > MAX_SECONDS:
        raise gr.Error(f"This demo accepts up to {MAX_SECONDS / 60:.0f} minutes of audio; yours is {len(wav) / sr / 60:.1f} minutes.")
    progress(0.1, desc="running the model")
    post = d.posteriors(wav, progress=lambda f: progress(0.1 + 0.8 * f))
    from ildiar.decode import decode, to_segments
    labels, chosen = decode(post, langs or None)
    segs = to_segments(labels)
    name = os.path.splitext(os.path.basename(audio_path))[0]
    out = tempfile.mkdtemp(prefix="ildiar_")
    rttm = write_rttm(segs, name, os.path.join(out, f"{name}.rttm"))
    t0 = max(0.0, float(win_start))
    png = plot_diarization(wav, segs, os.path.join(out, f"{name}_languages.png"), window=(t0, t0 + float(win_len)), title=name)
    files = split_by_language(wav, segs, out, name)
    from ildiar.classes import CODE2ID, ID2CODE
    order = ["en"] + [ID2CODE[c] for c in chosen]
    outs = [files.get(c) for c in order[:3]] + [None] * (3 - len(order[:3]))
    summary = "Detected languages: English" + "".join(f", {LANG_NAMES[ID2CODE[c]]}" for c in chosen) if chosen else "No Indian language detected."
    with open(rttm, encoding="utf-8") as f:
        text = f.read()
    return summary, rttm, text, png, *outs


choices = ["auto"] + LANGS
with gr.Blocks(title="Indic language diarization") as demo:
    gr.Markdown(
        "# Language diarization: English + up to two Indian languages\n"
        "Upload a recording that mixes **English** and **one or two** of: " + ", ".join(LANG_NAMES[l] for l in LANGS) + ".\n"
        "You get (a) an RTTM file, (b) the waveform coloured by language, and (c) one pooled audio file per language.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(type="filepath", label="Input speech (wav / mp3 / m4a / flac ...)")
            with gr.Row():
                l1 = gr.Dropdown(choices, value="auto", label="Indian language 1")
                l2 = gr.Dropdown(choices, value="auto", label="Indian language 2")
            gr.Markdown("Leave both on *auto* to detect the languages (less accurate than naming them).")
            with gr.Row():
                ws = gr.Number(value=0, label="Picture: start (s)")
                wl = gr.Number(value=20, label="Picture: length (s)")
            go = gr.Button("Diarize", variant="primary")
        with gr.Column():
            summary = gr.Textbox(label="Result", interactive=False)
            img = gr.Image(label="Language-coloured waveform", type="filepath")
            rttm_file = gr.File(label="RTTM file")
            rttm_text = gr.Textbox(label="RTTM", lines=8, interactive=False)
    with gr.Row():
        a1 = gr.Audio(label="Pooled audio: English", type="filepath")
        a2 = gr.Audio(label="Pooled audio: Indian language 1", type="filepath")
        a3 = gr.Audio(label="Pooled audio: Indian language 2", type="filepath")
    go.click(run, [audio, l1, l2, ws, wl], [summary, rttm_file, rttm_text, img, a1, a2, a3])
    gr.Markdown("Research prototype: trained on synthetic code-switched speech; accuracy on real conversations is limited "
                "(about 26 % diarization error with the languages named, see the README).")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
