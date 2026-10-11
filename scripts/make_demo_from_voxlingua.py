"""Build a short three-language demonstration recording (English + two Indian languages) from VoxLingua107 clips.

The clips are cropped to speech, normalised, joined with short pauses and a little noise; a reference RTTM is written.
Only clips of videos in the held-out 'dev' split (crc32(video id) mod 100 < 10, never used for training) are used.

python scripts/make_demo_from_voxlingua.py --vox-dir /path/voxlingua --langs ta gu --out demo --seed 7
  --vox-dir must contain raw_<lang>/<lang>/*.wav (and raw_en/en/*.wav), i.e. the unzipped VoxLingua107 archives.
"""
import argparse
import glob
import os
import sys
import zlib

import numpy as np
import soundfile as sf

SR = 16000
FR = 320


def is_dev(path):
    return zlib.crc32(os.path.basename(path).split("__U__")[0].encode()) % 100 < 10


def speech_crop(x, dur, rng):
    """random crop of ~dur seconds, trimmed to energy-detected speech"""
    n = len(x) // FR
    f = x[: n * FR].reshape(n, FR)
    db = 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-8)
    m = db > np.percentile(db, 95) - 28
    nf = int(dur / 0.02)
    if n < nf + 5:
        return None
    for _ in range(30):
        s = rng.randint(0, n - nf)
        mm = m[s:s + nf]
        if mm.mean() < 0.7:
            continue
        idx = np.where(mm)[0]
        a, b = s + idx[0], s + idx[-1] + 1
        seg = x[a * FR: b * FR]
        return seg * (10 ** (-23 / 20) / (np.sqrt((seg[m[a:b].repeat(FR)[: len(seg)]] ** 2).mean()) + 1e-8))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vox-dir", required=True)
    ap.add_argument("--langs", nargs=2, required=True, help="two Indian language codes, e.g. ta gu")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--extra-dir", default=None, help="optional directory with replacement clips (raw_extra/<lang>/)")
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    os.makedirs(a.out, exist_ok=True)

    clips = {}
    for l in ["en"] + a.langs:
        d = {os.path.basename(p): p for p in glob.glob(f"{a.vox_dir}/raw_{l}/{l}/*.wav")}
        if a.extra_dir:
            d.update({os.path.basename(p): p for p in glob.glob(f"{a.extra_dir}/{l}/*.wav")})
        clips[l] = [p for p in sorted(d.values()) if is_dev(p)]
        print(l, len(clips[l]), "held-out clips")

    order = [a.langs[0], a.langs[1], "en"] * 4          # L1 L2 E L1 L2 E ...
    audio, t, ref = [], 0.0, []
    for l in order:
        if t >= a.duration:
            break
        seg = None
        while seg is None:
            x, sr = sf.read(clips[l][rng.randint(len(clips[l]))], dtype="float32")
            seg = speech_crop(x, rng.uniform(2.0, 4.5), rng)
        gap = rng.uniform(0.2, 0.5)
        audio.append(np.zeros(int(gap * SR), np.float32))
        t += gap
        audio.append(seg)
        ref.append((t, len(seg) / SR, l))
        t += len(seg) / SR
    x = np.concatenate(audio)
    x = x + rng.randn(len(x)).astype(np.float32) * np.sqrt((x ** 2).mean()) * 10 ** (-30 / 20)
    x = x / np.abs(x).max() * 0.8
    wav = os.path.join(a.out, "demo_input.wav")
    sf.write(wav, x, SR, subtype="PCM_16")
    with open(os.path.join(a.out, "demo_reference.rttm"), "w") as f:
        for s, d, l in ref:
            f.write(f"LANGUAGE demo_input 1 {s:.3f} {d:.3f} <NA> <NA> {l.upper()} <NA> <NA>\n")
    print(f"wrote {wav} ({len(x) / SR:.1f} s, {len(ref)} turns)")


if __name__ == "__main__":
    main()
