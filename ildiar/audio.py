"""Audio loading: any common format -> 16 kHz mono float32."""
import shutil
import subprocess
from math import gcd

import numpy as np

SR = 16000


def load_audio(path):
    """Return (waveform float32 [N], 16000). Uses soundfile when it can read the file, otherwise ffmpeg
    (needed e.g. for mp3 / m4a / mp4 news clips on older soundfile builds)."""
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"cannot read {path!r}: soundfile failed and ffmpeg is not installed")
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32).copy(), SR
    if sr != SR:
        from scipy.signal import resample_poly
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x.astype(np.float32), SR
