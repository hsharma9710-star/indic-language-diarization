"""Language diarization + English second opinion.

The diarizer sometimes labels Indian-accented English as the Indian language (e.g. Hindi news). This script adds an
English-evidence check: an English CTC speech recogniser (facebook/wav2vec2-base-960h) is run on the audio, and a 200 ms
bin that is currently labelled as an Indian language is switched to English when enough of the neighbouring ~4 s is
recognised as real English words.

    python english_boost.py talk.mp3 -o out --languages hi

Settings (min word length 4, 4 s smoothing, threshold 0.25) were chosen on 6 real recordings and checked on 6 others.
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

STOP = set("to of is in on it as at be by he we so or an do go no my me up if am us the and but for you are was not can had has his her him she who did get got let say see too two one out our all any how why now yes".split())
WORDS = set(open(os.path.join(HERE, "english_words.txt"), encoding="utf-8").read().split())


def _load_asr():
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval()
    return proc, model


@torch.no_grad()
def recognised_words(wav, proc, model, chunk_s=20.0):
    vocab = {v: k for k, v in proc.tokenizer.get_vocab().items()}
    n = int(chunk_s * 16000)
    words, off = [], 0.0
    for a in range(0, len(wav), n):
        x = wav[a:a + n]
        if len(x) < 1600:
            continue
        x = (x - x.mean()) / (x.std() + 1e-7)
        ids = model(torch.from_numpy(x.astype(np.float32))[None]).logits[0].argmax(-1).tolist()
        cur, start, last, prev = "", None, None, -1
        for t, i in enumerate(ids):
            if i == prev:
                continue
            prev = i
            ch = vocab[i]
            if ch in ("<pad>", "<s>", "</s>", "<unk>"):
                continue
            if ch == "|":
                if cur:
                    words.append((a / 16000 + start * 0.02, a / 16000 + (last + 1) * 0.02, cur))
                    cur = ""
                continue
            if not cur:
                start = t
            cur += ch.lower()
            last = t
        if cur:
            words.append((a / 16000 + start * 0.02, a / 16000 + (last + 1) * 0.02, cur))
    return words


def english_evidence(words, n_bins, min_len=4, smooth_bins=20):
    v = np.zeros(n_bins)
    for s, e, w in words:
        if (w in WORDS and len(w) >= min_len) or w in STOP:
            for b in range(int(s / BIN), min(int(np.ceil(e / BIN)), n_bins)):
                v[b] += max(0.0, min(e, (b + 1) * BIN) - max(s, b * BIN)) / BIN
    return np.convolve(np.minimum(v, 1.0), np.ones(smooth_bins) / smooth_bins, mode="same")


def boost(labels, evidence, tau=0.25):
    out = labels.copy()
    out[(labels != SIL) & (evidence >= tau)] = ENG
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default="out_boost")
    ap.add_argument("--languages", nargs="+", help="Indian language(s) present, e.g. hi")
    ap.add_argument("--weights", help="path to diarizer.pt (default: as for the main tool)")
    ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--plot-window", nargs=2, type=float, default=[0.0, 30.0])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = Diarizer(find_weights(a.weights), "cuda" if torch.cuda.is_available() else "cpu")
    wav, _ = load_audio(a.audio)
    wav = np.asarray(wav, dtype=np.float32)
    name = os.path.splitext(os.path.basename(a.audio))[0]
    post = d.posteriors(wav)
    base, chosen = decode(post, a.languages)
    proc, model = _load_asr()
    words = recognised_words(wav, proc, model)
    ev = english_evidence(words, len(post))
    new = boost(base, ev, a.tau)
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
