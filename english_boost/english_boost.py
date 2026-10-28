"""Language diarization + English check with Whisper-tiny language detection.

The diarizer sometimes labels Indian-accented English as the Indian language (e.g. Hindi news). Here Whisper-tiny's language
detector is run on 4 s windows (one every 2 s); a speech bin that the diarizer calls an Indian language is switched to English
where the interpolated English probability is at least 0.3.

    python english_boost.py talk.mp3 -o out --languages hi

Settings (4 s windows, 2 s hop, threshold 0.3) were chosen on 6 real recordings and checked on 6 others.
The first run downloads openai/whisper-tiny (about 150 MB) from Hugging Face.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # repository root
from ildiar.classes import ENG, SIL                       # noqa: E402
from ildiar.decode import BIN, decode, to_segments        # noqa: E402
from ildiar.engine import Diarizer                        # noqa: E402
from ildiar.weights import find_weights                   # noqa: E402
from ildiar.audio import load_audio                       # noqa: E402
from ildiar.outputs import plot_diarization, split_by_language, write_rttm   # noqa: E402

NAME = "openai/whisper-tiny"
WIN_S, HOP_S, THETA = 4.0, 2.0, 0.3


@torch.no_grad()
def english_probability(wav, n_bins, batch=8):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(NAME)
    model = WhisperForConditionalGeneration.from_pretrained(NAME).eval()
    lang_to_id = model.generation_config.lang_to_id
    langs = list(lang_to_id.keys())
    ids = torch.tensor([lang_to_id[l] for l in langs])
    en = langs.index("<|en|>")
    sot = model.generation_config.decoder_start_token_id
    n, w, h = len(wav), int(WIN_S * 16000), int(HOP_S * 16000)
    starts = list(range(0, max(n - w, 0) + 1, h)) or [0]
    pen = []
    for i in range(0, len(starts), batch):
        chunk = [wav[s:s + w] for s in starts[i:i + batch]]
        feats = proc.feature_extractor(chunk, sampling_rate=16000, return_tensors="pt").input_features
        enc = model.model.encoder(feats).last_hidden_state
        dec = model.model.decoder(input_ids=torch.full((len(chunk), 1), sot, dtype=torch.long), encoder_hidden_states=enc).last_hidden_state
        pen.append(torch.softmax(model.proj_out(dec)[:, 0, :][:, ids], -1)[:, en].numpy())
    pen = np.concatenate(pen)
    centres = np.array([(s + w / 2) / 16000 for s in starts])
    return np.interp((np.arange(n_bins) + 0.5) * BIN, centres, pen)


def boost(labels, pen, theta=THETA):
    out = labels.copy()
    out[(labels != SIL) & (labels != ENG) & (pen >= theta)] = ENG
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="out_boost")
    ap.add_argument("--languages", nargs="+", help="Indian language(s) present, e.g. hi")
    ap.add_argument("--weights", help="path to diarizer.pt (default: as for the main tool)")
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--plot-window", nargs=2, type=float, default=[0.0, 30.0])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = Diarizer(find_weights(a.weights), "cuda" if torch.cuda.is_available() else "cpu")
    wav, _ = load_audio(a.audio)
    wav = np.asarray(wav, dtype=np.float32)
    name = os.path.splitext(os.path.basename(a.audio))[0]
    post = d.posteriors(wav)
    base, _ = decode(post, a.languages)
    new = boost(base, english_probability(wav, len(post)), a.theta)
    for tag, lab in (("before", base), ("after", new)):
        segs = to_segments(lab)
        write_rttm(segs, name, os.path.join(a.out, f"{name}_{tag}.rttm"))
        plot_diarization(wav, segs, os.path.join(a.out, f"{name}_{tag}.png"), window=tuple(a.plot_window), title=f"{name} ({tag})")
        if tag == "after":
            split_by_language(wav, segs, a.out, name + "_after")
    speech = base != SIL
    print(f"English before: {100 * (base[speech] == ENG).mean():.1f} % of speech;  after: {100 * (new[speech] == ENG).mean():.1f} %")
    print("outputs in", a.out)


if __name__ == "__main__":
    main()
