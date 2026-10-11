"""Diarizer: audio -> per-bin class posteriors -> language segments."""
import os

import numpy as np
import torch

from .audio import SR, load_audio
from .classes import SYMS
from .decode import decode, to_segments
from .model import W2V_EA, W2V_ES

CHUNK_S = 30.0          # the network was trained on 20-40 s utterances and evaluated on 30 s chunks
MIN_TAIL_S = 2.0
BIN_S = 0.2


class Diarizer:
    def __init__(self, weights, device=None):
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        cfg = Wav2Vec2Config.from_dict(ck["encoder_config"])
        self.encoder = Wav2Vec2Model(cfg)
        self.encoder.load_state_dict(ck["encoder"], strict=False)       # masked_spec_embed is not needed at inference
        self.encoder.to(self.device).eval()
        M = W2V_ES if ck["pooling"] == "stat" else W2V_EA
        self.head = M(n_lang=len(SYMS))
        self.head.load_state_dict(ck["head"])
        self.head.to(self.device).eval()
        self.meta = ck.get("meta", {})

    @torch.no_grad()
    def posteriors(self, wav, progress=None):
        """wav float32 [N] at 16 kHz -> [n_bins, 15] posteriors (200 ms bins)."""
        n = len(wav)
        chunk = int(CHUNK_S * SR)
        starts = list(range(0, max(n, 1), chunk))
        if len(starts) > 1 and n - starts[-1] < MIN_TAIL_S * SR:          # merge a very short tail into the previous chunk
            starts.pop()
        out = []
        for k, a in enumerate(starts):
            b = n if k == len(starts) - 1 else a + chunk
            x = torch.from_numpy(wav[a:b]).float().unsqueeze(0).to(self.device)
            want = int(np.ceil((b - a) / (BIN_S * SR)))              # bins this chunk should cover in time
            if x.shape[1] < int(0.5 * SR):
                out.append(np.tile(np.eye(len(SYMS), dtype=np.float32)[0], (want, 1)))   # too short: call it silence
                continue
            feats = self.encoder(x).last_hidden_state                         # [1, T, 768] at 20 ms
            logits, _ = self.head(feats)
            p = torch.softmax(logits[0].float(), -1).cpu().numpy()
            if len(p) < want:                                                 # T // 10 drops the last <200 ms: keep timing exact
                p = np.concatenate([p, np.tile(p[-1:], (want - len(p), 1))], 0)
            out.append(p[:want])
            if progress is not None:
                progress((k + 1) / len(starts))
        return np.concatenate(out, 0) if out else np.zeros((0, len(SYMS)), dtype=np.float32)

    def diarize(self, path, languages=None, max_indic=2, smooth=1, min_dur=0.0, progress=None):
        wav, _ = load_audio(path)
        post = self.posteriors(wav, progress)
        labels, chosen = decode(post, languages, max_indic, smooth)
        segs = to_segments(labels, min_dur=min_dur)
        return dict(wav=wav, post=post, labels=labels, segments=segs, languages=chosen,
                    duration=len(wav) / SR, name=os.path.splitext(os.path.basename(str(path)))[0])
