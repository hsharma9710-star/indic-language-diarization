"""Command line: audio -> RTTM (+ coloured picture + per-language pooled audio).

    python -m ildiar.cli talk.wav -o out/
    python -m ildiar.cli talk.mp3 -o out/ --languages ml kn --plot-window 30 50
"""
import argparse
import os
import sys
import time

from .classes import SUPPORTED
from .engine import Diarizer
from .outputs import plot_diarization, split_by_language, write_rttm
from .weights import find_weights


def main(argv=None):
    ap = argparse.ArgumentParser(description="Language diarization of speech with English and up to two Indian languages.")
    ap.add_argument("audio", nargs="+", help="input audio file(s): wav, flac, mp3, m4a, mp4, ... (ffmpeg is used for non-wav)")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("--languages", nargs="+", choices=[l for l in SUPPORTED if l != "en"], metavar="LANG",
                    help="the Indian language(s) present (1 or 2 ISO codes, e.g. ml kn); if omitted they are detected")
    ap.add_argument("--weights", help="path to diarizer.pt")
    ap.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--min-dur", type=float, default=0.0, help="absorb segments shorter than this many seconds into the previous one (default 0 = off)")
    ap.add_argument("--smooth", type=int, default=1, help="majority filter length in 200 ms bins (default 1 = off; 5 gives smoother but, on development data, less accurate output)")
    ap.add_argument("--plot-window", nargs=2, type=float, default=[0.0, 20.0], metavar=("START", "END"),
                    help="seconds of audio shown in the coloured picture (default 0 20)")
    ap.add_argument("--rttm-codes", choices=["iso", "family"], default="iso", help="'family' writes Kannada as KA")
    ap.add_argument("--no-split", action="store_true", help="do not write the pooled per-language audio files")
    a = ap.parse_args(argv)
    if a.languages and len(a.languages) > 2:
        ap.error("at most two Indian languages (plus English) are supported")

    d = Diarizer(find_weights(a.weights), a.device)
    os.makedirs(a.out, exist_ok=True)
    for path in a.audio:
        t = time.time()
        r = d.diarize(path, a.languages, smooth=a.smooth, min_dur=a.min_dur)
        name = r["name"]
        rttm = write_rttm(r["segments"], name, os.path.join(a.out, f"{name}.rttm"), a.rttm_codes)
        png = plot_diarization(r["wav"], r["segments"], os.path.join(a.out, f"{name}_languages.png"),
                               window=tuple(a.plot_window), title=name)
        files = {}
        if not a.no_split:
            files = split_by_language(r["wav"], r["segments"], a.out, name)
        print(f"{name}: {r['duration']:.1f} s audio, {len(r['segments'])} segments in {time.time() - t:.1f} s")
        print("  RTTM  ", rttm)
        print("  image ", png)
        for k, v in files.items():
            print(f"  audio  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
