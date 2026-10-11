"""Quantize the ONNX model to int8 (MatMul weights) and compare DER fp32 vs int8 on real recordings (first N seconds of each).

python quantize_and_eval.py --fp32 web_model/model_fp32.onnx --audio-dir A --rttm-dir R --files SID03b SID07a ... --seconds 600
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from evaluate_dir import bins_of, read_ref
from ildiar.audio import load_audio
from ildiar.classes import ENG, ID2CODE, SIL
from ildiar.decode import BIN, decode

SR = 16000


def posteriors(sess, wav, chunk=30 * SR):
    n = len(wav)
    starts = list(range(0, n, chunk))
    if len(starts) > 1 and n - starts[-1] < 2 * SR:
        starts.pop()
    out = []
    for k, a in enumerate(starts):
        b = n if k == len(starts) - 1 else a + chunk
        want = int(np.ceil((b - a) / (BIN * SR)))
        p = sess.run(None, {"wav": wav[a:b][None].astype(np.float32)})[0][0]
        if len(p) < want:
            p = np.concatenate([p, np.tile(p[-1:], (want - len(p), 1))], 0)
        out.append(p[:want])
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--rttm-dir", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--seconds", type=float, default=600)
    a = ap.parse_args()

    int8 = a.fp32.replace("fp32", "int8")
    quantize_dynamic(a.fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    print("fp32 %.0f MB   int8 %.0f MB" % (os.path.getsize(a.fp32) / 1e6, os.path.getsize(int8) / 1e6), flush=True)

    sessions = {"fp32": ort.InferenceSession(a.fp32, providers=["CPUExecutionProvider"]),
                "int8": ort.InferenceSession(int8, providers=["CPUExecutionProvider"])}
    agg = {k: [0, 0, 0, 0] for k in sessions}       # errors named, speech named, errors auto, speech auto
    agree = []
    for name in a.files:
        wav, _ = load_audio(os.path.join(a.audio_dir, name + ".wav"))
        wav = wav[: int(a.seconds * SR)]
        segs = read_ref(os.path.join(a.rttm_dir, name + ".rttm"))
        posts = {k: posteriors(s, wav) for k, s in sessions.items()}
        ref = bins_of(segs, len(posts["fp32"]))
        sp = ref != SIL
        indic = sorted({c for _, _, c in segs if c != ENG})
        row = []
        for k, p in posts.items():
            lab, _ = decode(p, [ID2CODE[c] for c in indic])
            err = ((lab != ref) & sp).sum() + ((ref == SIL) & (lab != SIL)).sum()
            agg[k][0] += int(err)
            agg[k][1] += int(sp.sum())
            lab2, ch = decode(p, None)
            err2 = ((lab2 != ref) & sp).sum() + ((ref == SIL) & (lab2 != SIL)).sum()
            agg[k][2] += int(err2)
            agg[k][3] += int(sp.sum())
            row.append(f"{k}: given {err / max(sp.sum(), 1) * 100:.1f}% auto {err2 / max(sp.sum(), 1) * 100:.1f}%")
        agree.append((posts["fp32"].argmax(1) == posts["int8"].argmax(1)).mean())
        print(name, " | ".join(row), flush=True)
    for k, v in agg.items():
        print(f"POOLED {k}: DER languages-named {v[0] / v[1] * 100:.1f} %   auto {v[2] / v[3] * 100:.1f} %")
    print("frame argmax agreement fp32 vs int8: %.1f %%" % (np.mean(agree) * 100))


if __name__ == "__main__":
    main()
