"""Export the whole model (waveform -> per-200 ms class posteriors) to one ONNX graph, check it against PyTorch,
and write fp32 / int8 versions.

python export_onnx.py --weights weights/diarizer.pt --out web_model --audio some.wav
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ildiar.audio import load_audio
from ildiar.classes import SYMS
from ildiar.model import W2V_EA, W2V_ES


class Full(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, wav):                       # wav: [1, N] float32, 16 kHz
        feats = self.encoder(wav).last_hidden_state
        logits, _ = self.head(feats)
        return torch.softmax(logits, dim=-1)      # [1, N//320//10, 15]


def load(weights):
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    cfg = Wav2Vec2Config.from_dict(ck["encoder_config"])
    cfg._attn_implementation = "eager"
    enc = Wav2Vec2Model(cfg)
    enc.load_state_dict(ck["encoder"], strict=False)
    M = W2V_ES if ck["pooling"] == "stat" else W2V_EA
    head = M(n_lang=len(SYMS))
    head.load_state_dict(ck["head"])
    return Full(enc.eval(), head.eval()).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", nargs="+", required=True, help="audio files used to compare PyTorch and ONNX outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = load(a.weights)
    dummy = torch.randn(1, 16000 * 12) * 0.05
    path = os.path.join(a.out, "model_fp32.onnx")
    with torch.no_grad():
        torch.onnx.export(m, (dummy,), path, input_names=["wav"], output_names=["post"],
                          dynamic_axes={"wav": {1: "n"}, "post": {1: "s"}}, opset_version=17, dynamo=False)
    print("exported", path, os.path.getsize(path) / 1e6, "MB")

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for f in a.audio:
        wav, _ = load_audio(f)
        wav = wav[: 16000 * 30]
        x = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            ref = m(x).numpy()[0]
        out = sess.run(None, {"wav": wav[None].astype(np.float32)})[0][0]
        n = min(len(ref), len(out))
        print(f"{os.path.basename(f)}: bins torch {len(ref)} onnx {len(out)}  max|diff| {np.abs(ref[:n] - out[:n]).max():.2e}  "
              f"argmax agreement {(ref[:n].argmax(1) == out[:n].argmax(1)).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
