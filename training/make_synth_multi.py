import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Multilingual synthetic English + <Indian language> code-switched utterances from VoxLingua107.

  python make_synth_multi.py <out_name> <lang codes comma-sep> <train_hours> <dev_hours> <seed>

Every utterance mixes English with ONE (50 %) or TWO (50 %) different Indian languages (so at most 3 languages), matching the target use.
Labels:  labels.tsv   language-specific tag per 200 ms bin (S, E, one letter per Indian language)  -> Stage-2 multilingual fine-tuning
         labels3.tsv  the same with every Indian language collapsed to N                         -> Stage 3-5 (S / N / E classifier)
Turn / gap / VAD / noise design is identical to experimental_wav2vec2/code/make_synth.py.
"""
import glob, os, sys, json, zlib
import numpy as np, soundfile as sf
from multiprocessing import Pool

NAME = sys.argv[1]; LANGS = sys.argv[2].split(","); TRAIN_H = float(sys.argv[3]); DEV_H = float(sys.argv[4]); SEED = int(sys.argv[5])
ROOT = os.environ.get("ILDIAR_ROOT", ".")
VOX = os.environ.get("VOXLINGUA_DIR", ROOT + "/voxlingua")
OUT = ROOT + "/synthetic_data/" + NAME
TAG = {"as": "A", "bn": "B", "gu": "G", "hi": "H", "kn": "K", "ml": "M", "mr": "R", "ne": "P", "pa": "J", "sa": "X", "sd": "D",
       "ta": "T", "te": "L", "ur": "U"}
SR = 16000; FR = 320; BINF = 10
SYMS = ["S", "E"] + [TAG[l] for l in LANGS]          # class symbols: silence, English, one per Indian language
CODE = {"en": 1}
CODE.update({l: 2 + i for i, l in enumerate(LANGS)})


def vid_of(p): return os.path.basename(p).split("__U__")[0]
def is_dev(v): return zlib.crc32(v.encode()) % 100 < 10


def clean_runs(m, close=8, minrun=5):
    m = m.copy()
    d = np.diff(np.concatenate([[1], m.astype(np.int8), [1]]))
    for s, e in zip(np.where(d == -1)[0], np.where(d == 1)[0]):
        if e - s < close and s > 0 and e < len(m): m[s:e] = True
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if e - s < minrun: m[s:e] = False
    return m


def vad_mask(x):
    n = len(x) // FR
    if n < 5: return np.zeros(n, bool)
    f = x[: n * FR].reshape(n, FR)
    db = 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-8)
    return clean_runs(db > np.percentile(db, 95) - 28.0)


def pink(n, rng):
    W = np.fft.rfft(rng.randn(n).astype(np.float32)); f = np.arange(len(W), dtype=np.float32); f[0] = 1
    p = np.fft.irfft(W / np.sqrt(f), n).astype(np.float32)
    return p / (p.std() + 1e-9)


def draw_crop(path, rng, dur):
    x, sr = sf.read(path, dtype="float32"); assert sr == SR
    if x.ndim > 1: x = x.mean(1)
    m = vad_mask(x); n = len(m); nf = min(int(dur / 0.02), n)
    if n < 25: return None
    for _ in range(12):
        s = rng.randint(0, n - nf + 1); mm = m[s:s + nf]
        if mm.mean() < 0.5: continue
        idx = np.where(mm)[0]; a, b = s + idx[0], s + idx[-1] + 1
        if (b - a) < 40: continue
        seg = x[a * FR: b * FR]; mask = m[a:b]
        sp = np.concatenate([seg[i * FR:(i + 1) * FR] for i in np.where(mask)[0]]) if mask.any() else seg
        seg = seg * (10 ** (-23 / 20) / (np.sqrt((sp ** 2).mean()) + 1e-8))
        return seg[: (b - a) * FR], mask
    return None


def make_utt(args):
    k, split, seed = args
    rng = np.random.RandomState(seed)
    # 50 %: English + 1 Indian language, 50 %: English + 2 different Indian languages (<= 3 languages per utterance, as in the target use)
    nl = 1 if rng.rand() < 0.5 else 2
    inds = list(rng.choice(LANGS, nl, replace=False))
    langs = inds + ["en"]
    target = rng.uniform(20, 40)
    audio, labs, turns = [], [], []
    cur = langs[rng.randint(len(langs))]; total = 0.0; first = True
    while total < target:
        if not first:
            if rng.rand() < 0.85:
                cur = [l for l in langs if l != cur][rng.randint(len(langs) - 1)]
            if rng.rand() < 0.7:
                ng = int(round(rng.uniform(0.1, 1.2) / 0.02))
                audio.append(np.zeros(ng * FR, np.float32)); labs.append(np.zeros(ng, np.int8)); total += ng * 0.02
        first = False; got = None
        for _ in range(8):
            path = CLIPS[split][cur][rng.randint(len(CLIPS[split][cur]))]
            got = draw_crop(path, rng, float(np.exp(rng.uniform(np.log(1.0), np.log(12.0)))))
            if got is not None: break
        if got is None: continue
        seg, mask = got
        audio.append(seg); labs.append(np.where(mask, CODE[cur], 0).astype(np.int8)); total += len(mask) * 0.02
        turns.append([cur, os.path.basename(path), round(len(mask) * 0.02, 2)])
    x = np.concatenate(audio); lab = np.concatenate(labs)
    n = min(len(lab), len(x) // FR); x, lab = x[: n * FR], lab[:n]
    sig = np.sqrt((x ** 2).mean()) + 1e-8
    noise = pink(len(x), rng) if rng.rand() < 0.5 else rng.randn(len(x)).astype(np.float32)
    x = x + noise * sig * 10 ** (-rng.uniform(15, 35) / 20)
    x = x / (np.abs(x).max() + 1e-6) * rng.uniform(0.3, 0.9)
    nb = n // BINF
    L = lab[: nb * BINF].reshape(nb, BINF)
    ids = np.stack([(L == c).sum(1) for c in range(len(SYMS))], 1).argmax(1)
    s_lang = "".join(np.array(SYMS)[ids])
    name = "%s_%s_%05d" % (NAME, split, k)
    sf.write("%s/%s/wav/%s.wav" % (OUT, split, name), x[: nb * BINF * FR], SR, subtype="PCM_16")
    return name, s_lang, s_lang, turns, "+".join(inds)


CLIPS = {}


def main():
    en = sorted(glob.glob("%s/raw_en/en/*.wav" % VOX))
    for split, flag in (("train", False), ("dev", True)):
        CLIPS[split] = {"en": [p for p in en if is_dev(vid_of(p)) == flag]}
        for l in LANGS:
            CLIPS[split][l] = [p for p in sorted(glob.glob("%s/raw_%s/%s/*.wav" % (VOX, l, l))) if is_dev(vid_of(p)) == flag]
        print(split, {k: len(v) for k, v in CLIPS[split].items()}, flush=True)
        os.makedirs("%s/%s/wav" % (OUT, split), exist_ok=True)
    for split, hours in (("train", TRAIN_H), ("dev", DEV_H)):
        n = int(hours * 3600 / 30.0)
        jobs = [(k, split, SEED * 1000003 + (0 if split == "train" else 500000) + k) for k in range(n)]
        cnt = {}
        with Pool(16) as p, open("%s/%s/labels.tsv" % (OUT, split), "w") as f1, open("%s/%s/labels3.tsv" % (OUT, split), "w") as f3, \
                open("%s/%s/turns.jsonl" % (OUT, split), "w") as fj:
            tot = 0
            for i, (name, s, s3, turns, nl) in enumerate(p.imap(make_utt, jobs, chunksize=4)):
                f1.write("%s\t%s\n" % (name, s)); f3.write("%s\t%s\n" % (name, s3)); fj.write(json.dumps({"utt": name, "lang": nl, "turns": turns}) + "\n")
                tot += len(s) * 0.2; cnt[nl] = cnt.get(nl, 0) + 1
                if i % 500 == 0: print(split, i, n, "%.1f h" % (tot / 3600), flush=True)
        print(split, "done: %d utts, %.2f h" % (n, tot / 3600), "per-language utterances", cnt, flush=True)


if __name__ == "__main__":
    main()
