"""From per-bin class posteriors to language segments.

Posteriors are [N_bins, 15] at 200 ms resolution (class order: classes.SYMS).
"""
import numpy as np

from .classes import ENG, ID2CODE, INDIC_IDS, SIL, CODE2ID

BIN = 0.2


def pick_languages(post, max_indic=2, ratio=0.1, min_share=0.05):
    """Detect which Indian languages are present in the file.

    Mass of a class = sum of its posterior over the bins that are more speech than silence. The strongest Indian language
    is kept if it holds at least `min_share` of the speech mass; a second one is kept only if its mass is at least `ratio`
    times the first one's and at least `min_share` of the speech mass (values chosen on synthetic development data,
    see scripts/sweep_decoder.py)."""
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


def majority_filter(labels, win=5):
    """Replace every bin by the most frequent class in a window of `win` bins (ties keep the centre)."""
    if win <= 1:
        return labels
    n_cls = int(labels.max()) + 1
    half = win // 2
    onehot = np.zeros((len(labels), n_cls), dtype=np.int32)
    onehot[np.arange(len(labels)), labels] = 1
    pad = np.pad(onehot, ((half, half), (0, 0)))
    csum = np.cumsum(np.vstack([np.zeros((1, n_cls), dtype=np.int32), pad]), axis=0)
    counts = csum[win:] - csum[:-win]
    out = counts.argmax(1)
    tie = counts.max(1) == counts[np.arange(len(labels)), labels]
    out[tie] = labels[tie]
    return out


def decode(post, languages=None, max_indic=2, smooth=1):
    """post: [N, 15].  languages: optional list of ISO codes of the Indian languages known to be present
    (1 or 2 of them); if None the languages are chosen automatically (top `max_indic`).
    Returns (labels [N] class ids, chosen Indian class ids)."""
    if languages:
        ids = [CODE2ID[l.lower()] for l in languages]
    else:
        ids = pick_languages(post, max_indic)
    allowed = [SIL, ENG] + ids
    sub = post[:, allowed]
    labels = np.array(allowed)[sub.argmax(1)]
    labels = majority_filter(labels, smooth)
    return labels, ids


def to_segments(labels, min_dur=0.0, keep_silence=False):
    """Merge equal neighbours. Returns [(start_s, dur_s, class_id)]. Segments shorter than `min_dur` are absorbed
    into the preceding segment (never creating a new language boundary)."""
    segs = []
    s = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            segs.append([s * BIN, (i - s) * BIN, int(labels[s])])
            s = i
    if min_dur > 0:
        merged = []
        for sg in segs:
            if merged and sg[1] < min_dur:
                merged[-1][1] += sg[1]
            else:
                merged.append(sg)
        segs = merged
    if not keep_silence:
        segs = [s for s in segs if s[2] != SIL]
    return [(a, b, c) for a, b, c in segs]


def code_of(class_id):
    return ID2CODE[class_id]
