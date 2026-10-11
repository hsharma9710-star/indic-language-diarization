"""Score a hypothesis RTTM against a reference RTTM on 200 ms bins (no collar).

python scripts/score_rttm.py --ref ref.rttm --hyp hyp.rttm

DER = (speech bins with the wrong / missed language + silence bins labelled as speech) / reference-speech bins.
Reference time not covered by any segment is silence. Language codes are compared case-insensitively; KA == KN.
"""
import argparse
from collections import Counter

import numpy as np

BIN = 0.2
ALIAS = {"KA": "KN"}


def read(path):
    segs = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) > 7 and p[0] == "LANGUAGE":
            lab = p[7].upper()
            segs.append((float(p[3]), float(p[3]) + float(p[4]), ALIAS.get(lab, lab)))
    return segs


def to_bins(segs, n):
    """Per 200 ms bin: the language covering most of the bin; silence where no segment covers more than half of it."""
    cov = {}
    for a, b, l in segs:
        arr = cov.setdefault(l, np.zeros(n))
        for i in range(int(a / BIN), min(int(np.ceil(b / BIN)), n)):
            arr[i] += max(0.0, min(b, (i + 1) * BIN) - max(a, i * BIN))
    lab = np.array(["SIL"] * n, dtype=object)
    if not cov:
        return lab
    names = list(cov)
    m = np.stack([cov[k] for k in names], 0)
    sil = np.maximum(BIN - m.sum(0), 0)
    pick = np.array(names, dtype=object)[m.argmax(0)]
    keep = m.max(0) > sil
    lab[keep] = pick[keep]
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", required=True)
    a = ap.parse_args()
    ref, hyp = read(a.ref), read(a.hyp)
    dur = max([e for _, e, _ in ref + hyp] + [BIN])
    n = int(np.ceil(dur / BIN))
    r, h = to_bins(ref, n), to_bins(hyp, n)
    sp = r != "SIL"
    err = ((r != h) & sp).sum() + ((r == "SIL") & (h != "SIL")).sum()
    der = err / max(sp.sum(), 1)
    print(f"bins {n}  reference speech {sp.sum() * BIN:.0f} s  DER {der * 100:.1f} %  accuracy {(r == h).mean() * 100:.1f} %")
    print("reference classes :", dict(Counter(r)))
    print("hypothesis classes:", dict(Counter(h)))


if __name__ == "__main__":
    main()
