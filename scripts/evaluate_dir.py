"""Run the tool on a directory of audio files that have reference RTTMs and report DER (200 ms bins, no collar).

python scripts/evaluate_dir.py --weights weights/diarizer.pt --audio-dir audio --rttm-dir rttms [--out results.json]

Two modes are scored:
  auto  : the Indian languages are detected from the audio (the default behaviour of the tool)
  given : the Indian language(s) are taken from the reference RTTM and passed to the tool (--languages)
DER = (speech bins with the wrong / missed language + silence bins labelled as speech) / reference-speech bins, pooled over
all files; reference time not covered by a segment is silence.  KA in a reference RTTM is read as Kannada.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ildiar.audio import load_audio
from ildiar.classes import CODE2ID, ENG, ID2CODE, RTTM_ALIASES, SIL
from ildiar.decode import BIN, decode
from ildiar.engine import Diarizer


def read_ref(path):
    segs = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) > 7 and p[0] == "LANGUAGE":
            code = RTTM_ALIASES.get(p[7].upper(), p[7]).lower()
            if code in CODE2ID:
                segs.append((float(p[3]), float(p[3]) + float(p[4]), CODE2ID[code]))
    return segs


def bins_of(segs, n):
    """Per 200 ms bin: the class that covers most of the bin, silence covering whatever no segment covers."""
    cov = {}
    for a, b, c in segs:
        arr = cov.setdefault(c, np.zeros(n))
        for i in range(int(a / BIN), min(int(np.ceil(b / BIN)), n)):
            arr[i] += max(0.0, min(b, (i + 1) * BIN) - max(a, i * BIN))
    classes = list(cov)
    if not classes:
        return np.zeros(n, dtype=int)
    m = np.stack([cov[c] for c in classes], 0)
    sil = np.maximum(BIN - m.sum(0), 0)
    lab = np.array(classes)[m.argmax(0)]
    lab[sil >= m.max(0)] = SIL
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--rttm-dir", required=True)
    ap.add_argument("--out")
    ap.add_argument("--device")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    d = Diarizer(a.weights, a.device)
    files = sorted(glob.glob(os.path.join(a.audio_dir, "*.wav")))
    if a.limit:
        files = files[: a.limit]
    agg = {m: [0, 0] for m in ("auto", "given")}
    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        ref_path = os.path.join(a.rttm_dir, name + ".rttm")
        if not os.path.exists(ref_path):
            continue
        ref_segs = read_ref(ref_path)
        wav, _ = load_audio(f)
        post = d.posteriors(wav)
        ref = bins_of(ref_segs, len(post))
        sp = ref != SIL
        indic = sorted({c for _, _, c in ref_segs if c != ENG})
        row = {"file": name, "ref_languages": [ID2CODE[c] for c in indic], "ref_speech_s": float(sp.sum() * BIN)}
        for mode, langs in (("auto", None), ("given", [ID2CODE[c] for c in indic])):
            lab, chosen = decode(post, langs)
            err = ((lab != ref) & sp).sum() + ((ref == SIL) & (lab != SIL)).sum()
            agg[mode][0] += int(err)
            agg[mode][1] += int(sp.sum())
            row[mode] = {"der": float(err / max(sp.sum(), 1)), "languages": [ID2CODE[c] for c in chosen]}
        rows.append(row)
        print(f"{name:8s} ref={row['ref_languages']} auto DER {row['auto']['der'] * 100:5.1f} ({row['auto']['languages']}) "
              f"given DER {row['given']['der'] * 100:5.1f}", flush=True)
    summary = {m: agg[m][0] / max(agg[m][1], 1) for m in agg}
    set_ok = np.mean([sorted(r["auto"]["languages"]) == sorted(r["ref_languages"]) for r in rows]) if rows else 0
    print(f"\nPOOLED DER  auto {summary['auto'] * 100:.1f} %   given {summary['given'] * 100:.1f} %   "
          f"(files {len(rows)}; language set detected exactly in {set_ok * 100:.0f} % of files)")
    if a.out:
        json.dump(dict(summary=summary, language_set_exact=float(set_ok), files=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
