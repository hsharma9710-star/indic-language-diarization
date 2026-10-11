"""Sweep decoder settings (second-language rule, smoothing) on audio with reference labels.

Used with the synthetic development set (utterances with 1 or 2 Indian languages) to choose the automatic language-detection rule
WITHOUT looking at the real test recordings.

python scripts/sweep_decoder.py --weights weights/diarizer.pt --wav-dir dev/wav --labels dev/labels.tsv --turns dev/turns.jsonl
  labels.tsv : <utt>\t<one symbol per 200 ms bin>   (symbols as in ildiar.classes.SYMS)
  turns.jsonl: one JSON per utterance with "lang" = "ml+ta" style string of the Indian languages present
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ildiar.audio import load_audio
from ildiar.classes import ENG, INDIC_IDS, SIL, SYMS
from ildiar.decode import majority_filter
from ildiar.engine import Diarizer


def pick(post, ratio, min_share, max_indic=2):
    speech = post[:, SIL] < 0.5
    if speech.sum() == 0:
        return []
    mass = post[speech].sum(0)
    total = mass[ENG] + mass[INDIC_IDS].sum()
    order = sorted(INDIC_IDS, key=lambda i: -mass[i])
    first = order[0]
    ids = [first] if mass[first] >= min_share * total else []
    for i in order[1:max_indic]:
        if ids and mass[i] >= ratio * mass[first] and mass[i] >= min_share * total:
            ids.append(i)
    return ids


def decode(post, ids, smooth):
    allowed = [SIL, ENG] + list(ids)
    lab = np.array(allowed)[post[:, allowed].argmax(1)]
    return majority_filter(lab, smooth)


def der(ref, hyp):
    n = min(len(ref), len(hyp))
    ref, hyp = ref[:n], hyp[:n]
    sp = ref != SIL
    return (((ref != hyp) & sp).sum() + ((ref == SIL) & (hyp != SIL)).sum()) / max(sp.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device")
    a = ap.parse_args()

    ref = {}
    for l in open(a.labels):
        u, s = l.rstrip("\n").split("\t")
        ref[u] = np.array([SYMS.index(c) for c in s])
    true_langs = {}
    for l in open(a.turns):
        d = json.loads(l)
        true_langs[d["utt"]] = set(x for x in d["lang"].split("+"))
    names = sorted(ref)
    if a.limit:
        names = names[: a.limit]
    d = Diarizer(a.weights, a.device)
    posts = {}
    for i, u in enumerate(names):
        wav, _ = load_audio(os.path.join(a.wav_dir, u + ".wav"))
        posts[u] = d.posteriors(wav)
        if i % 100 == 0:
            print("posteriors", i, len(names), flush=True)

    from ildiar.classes import ID2CODE
    print("\nratio  min_share  smooth | set-exact  |  DER %   (1-language / 2-language utterances DER)")
    results = []
    for ratio in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01):
        for ms in (0.0, 0.02, 0.05):
            for sm in (1, 5):
                ok = 0
                dr, d1, d2 = [], [], []
                for u in names:
                    ids = pick(posts[u], ratio, ms)
                    got = set(ID2CODE[i] for i in ids)
                    ok += got == true_langs[u]
                    e = der(ref[u], decode(posts[u], ids, sm))
                    dr.append(e)
                    (d1 if len(true_langs[u]) == 1 else d2).append(e)
                results.append((ratio, ms, sm, ok / len(names), np.mean(dr), np.mean(d1), np.mean(d2)))
                print(f"{ratio:4.2f}  {ms:5.2f}    {sm:3d}   |  {ok / len(names):.3f}   |  {np.mean(dr) * 100:5.1f}  ({np.mean(d1) * 100:5.1f} / {np.mean(d2) * 100:5.1f})")
    # oracle: the true language set is given
    for sm in (1, 5):
        e = [der(ref[u], decode(posts[u], [SYMS.index(x) if False else None for x in []] or [], sm)) for u in names[:0]]
    oracle = []
    from ildiar.classes import CODE2ID
    for u in names:
        ids = [CODE2ID[x] for x in true_langs[u]]
        oracle.append(der(ref[u], decode(posts[u], ids, 5)))
    print(f"\noracle (true languages given), smooth 5: DER {np.mean(oracle) * 100:.1f} %")


if __name__ == "__main__":
    main()
