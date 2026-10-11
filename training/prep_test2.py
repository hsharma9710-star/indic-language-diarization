import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""family_ld (real recordings + RTTMs) -> 16 kHz mono 30 s chunks + 200 ms label strings in the 15-class alphabet.

python prep_test2.py <family_ld dir with audio/ and rttms/> <out dir>
RTTM language codes (column 8) are mapped: EN->E, ML->M, KA/KN->K, HI->H, TA->T, TE->L, BN->B, GU->G, MR->R, PA->J, UR->U, AS->A, NE->P, SA->X.
Time not covered by any segment is silence (S). Recording pair = the non-English language(s) found in its RTTM.
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SRC, OUT = sys.argv[1], sys.argv[2]
WIN, BIN, SR = 30.0, 0.2, 16000
CODE = {"EN": "E", "ML": "M", "KA": "K", "KN": "K", "HI": "H", "TA": "T", "TE": "L", "BN": "B", "GU": "G", "MR": "R", "PA": "J",
        "UR": "U", "AS": "A", "NE": "P", "SA": "X"}
os.makedirs(OUT, exist_ok=True)
tsv = open(OUT + "/labels.tsv", "w")
meta = open(OUT + "/recordings.tsv", "w")
for wavp in sorted(glob.glob(SRC + "/audio/*.wav")):
    sid = os.path.splitext(os.path.basename(wavp))[0]
    cands = glob.glob(f"{SRC}/rttms/{sid}*.rttm") + glob.glob(f"{SRC}/rttms/{sid}*.RTTM")
    if not cands:
        print("NO RTTM for", sid, "- skipped", flush=True)
        continue
    segs = []
    for l in open(cands[0]):
        p = l.split()
        if len(p) > 7:
            segs.append((float(p[3]), float(p[3]) + float(p[4]), CODE.get(p[7].upper(), "?")))
    try:
        x, sr = sf.read(wavp, dtype="float32")
    except Exception as e:
        print("cannot read", sid, e, flush=True)
        continue
    if x.ndim > 1:
        x = x.mean(1)
    assert sr % SR == 0
    x = resample_poly(x, 1, sr // SR).astype(np.float32)
    dur = len(x) / SR
    nb_total = int(np.ceil(dur / BIN))
    syms = ["S"] + sorted({s[2] for s in segs})
    cov = np.zeros((len(syms), nb_total), dtype=np.float32)
    for a, b, lg in segs:
        k = syms.index(lg)
        for i in range(int(a / BIN), min(int(np.ceil(b / BIN)), nb_total)):
            lo, hi = max(a, i * BIN), min(b, (i + 1) * BIN)
            if hi > lo:
                cov[k, i] += hi - lo
    cov[0] = np.maximum(BIN - cov[1:].sum(0), 0)
    s_all = "".join(np.array(syms)[cov.argmax(0)])
    n = 0
    nb_win = int(WIN / BIN)
    os.makedirs(OUT + "/wav", exist_ok=True)
    for c0 in range(0, nb_total, nb_win):
        c1 = min(c0 + nb_win, nb_total)
        if (c1 - c0) * BIN < 2.0:
            continue
        name = f"{sid}_c{n:04d}"
        sf.write(f"{OUT}/wav/{name}.wav", x[int(c0 * BIN * SR): int(c1 * BIN * SR)], SR, subtype="PCM_16")
        tsv.write(f"{name}\t{s_all[c0:c1]}\n")
        n += 1
    langs = ",".join(sorted({s[2] for s in segs if s[2] != "E"}))
    meta.write(f"{sid}\t{langs}\t{dur:.0f}\t{n}\n")
    print(sid, "native:", langs, "dur %.0f" % dur, "chunks", n, "S %.1f%%" % (100 * s_all.count("S") / len(s_all)), flush=True)
tsv.close(); meta.close()
print("DONE")
