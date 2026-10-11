import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Extract frozen 768-d / 20 ms features for every wav in a directory (float16 .npy).
python extract_features.py --enc pre|<finetuned ckpt> --wav_dir D --out_dir O [--shard i/n]"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf
import torch

from hf_utils import frame_features, load_clsril, load_finetuned

ap = argparse.ArgumentParser()
ap.add_argument("--enc", required=True, help="'pre' = plain CLSRIL-23, otherwise a finetune_stage2.py checkpoint")
ap.add_argument("--wav_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--shard", default="0/1")
a = ap.parse_args()
dev = "cuda:0"
model = load_clsril(train=False).to(dev).eval() if a.enc == "pre" else load_finetuned(a.enc, dev)
os.makedirs(a.out_dir, exist_ok=True)
wavs = sorted(glob.glob(a.wav_dir + "/*.wav"))
i0, n = map(int, a.shard.split("/"))
wavs = wavs[i0::n]
for i, w in enumerate(wavs):
    o = os.path.join(a.out_dir, os.path.splitext(os.path.basename(w))[0] + ".npy")
    if os.path.exists(o):
        continue
    x, sr = sf.read(w, dtype="float32")
    assert sr == 16000, (w, sr)
    if x.ndim > 1:
        x = x.mean(1)
    np.save(o, frame_features(model, x, dev).astype(np.float16))
    if i % 500 == 0:
        print(i, len(wavs), flush=True)
print("done", a.out_dir)
