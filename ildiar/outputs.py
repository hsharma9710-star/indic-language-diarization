"""Writers: RTTM, coloured-waveform picture, per-language pooled audio."""
import os

import numpy as np

from .audio import SR
from .classes import ENG, ID2CODE, LANG_NAMES

# English is blue, the first / second Indian language pink / green (as in the project's figures)
COLOURS = {"en": "#3b82f6", "first": "#ec4899", "second": "#22c55e", "third": "#f59e0b"}


def rttm_lines(segments, name, code_style="iso"):
    """RTTM, one line per segment:  LANGUAGE <file> 1 <start> <dur> <NA> <NA> <LANG> <NA> <NA>"""
    out = []
    for start, dur, cid in segments:
        code = ID2CODE[cid].upper()
        if code_style == "family" and code == "KN":
            code = "KA"
        out.append(f"LANGUAGE {name} 1 {start:.3f} {dur:.3f} <NA> <NA> {code} <NA> <NA>")
    return out


def write_rttm(segments, name, path, code_style="iso"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rttm_lines(segments, name, code_style)) + ("\n" if segments else ""))
    return path


def language_colours(segments, languages):
    """language class id -> colour; Indian languages are coloured in order of first appearance."""
    cols = {ENG: COLOURS["en"]}
    order = []
    for _, _, c in segments:
        if c != ENG and c not in order:
            order.append(c)
    for i, c in enumerate(order):
        cols[c] = [COLOURS["first"], COLOURS["second"], COLOURS["third"]][min(i, 2)]
    return cols


def plot_diarization(wav, segments, path, window=(0.0, 20.0), title=None, languages=None):
    """Waveform of `window` = (start_s, end_s) coloured by language; silence stays grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    t0, t1 = window
    t1 = min(t1, len(wav) / SR)
    a, b = int(t0 * SR), int(t1 * SR)
    seg = wav[a:b]
    step = max(len(seg) // 4000, 1)                       # draw a min/max envelope, 4000 columns
    n = len(seg) // step
    blocks = seg[: n * step].reshape(n, step)
    lo, hi = blocks.min(1), blocks.max(1)
    t = t0 + (np.arange(n) + 0.5) * step / SR
    cols = language_colours(segments, languages)

    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.fill_between(t, lo, hi, color="#9ca3af", linewidth=0)
    used = []
    for s, d, c in segments:
        if s + d <= t0 or s >= t1:
            continue
        m = (t >= max(s, t0)) & (t <= min(s + d, t1))
        ax.fill_between(t[m], lo[m], hi[m], color=cols.get(c, "#6b7280"), linewidth=0)
        if c not in used:
            used.append(c)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    handles = [Patch(color=cols[c], label=LANG_NAMES[ID2CODE[c]]) for c in used]
    handles.append(Patch(color="#9ca3af", label="silence / no speech"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=len(handles), frameon=False)
    if title:
        ax.set_title(title, fontsize=9, pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def split_by_language(wav, segments, out_dir, name, fade_ms=10):
    """Pool all segments of one language into one audio file each: <name>_<LANG>.wav (segments in time order)."""
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    fade = int(fade_ms / 1000 * SR)
    paths = {}
    by = {}
    for s, d, c in segments:
        by.setdefault(c, []).append(wav[int(s * SR): int((s + d) * SR)].copy())
    for c, parts in by.items():
        for p in parts:                                   # short fades avoid clicks at the joins
            k = min(fade, len(p) // 2)
            if k > 0:
                ramp = np.linspace(0, 1, k, dtype=np.float32)
                p[:k] *= ramp
                p[-k:] *= ramp[::-1]
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        path = os.path.join(out_dir, f"{name}_{ID2CODE[c].upper()}.wav")
        sf.write(path, audio, SR, subtype="PCM_16")
        paths[ID2CODE[c]] = path
    return paths
