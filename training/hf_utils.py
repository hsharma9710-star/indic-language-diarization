import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Loading CLSRIL-23 (Hugging Face export of the pre-trained fairseq checkpoint, sha256 4f8be6d8...) and fine-tuned variants."""
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

import os
ROOT = os.environ.get("ILDIAR_ROOT", ".")
CLSRIL_DIR = os.environ.get("CLSRIL_DIR", ROOT + "/models/clsril23_hf")


def load_clsril(train=False, layerdrop=0.05, mask_time_prob=0.05):
    cfg = Wav2Vec2Config.from_pretrained(CLSRIL_DIR)
    cfg.layerdrop = layerdrop if train else 0.0
    cfg.apply_spec_augment = bool(train)
    cfg.mask_time_prob = mask_time_prob if train else 0.0
    cfg.mask_time_length = 10          # same mask length as the CLSRIL-23 pre-training (10 frames)
    cfg.mask_feature_prob = 0.0
    m = Wav2Vec2Model.from_pretrained(CLSRIL_DIR, config=cfg)
    return m


def load_finetuned(ckpt_path, device="cuda:0"):
    """ckpt from finetune_stage2.py: {'enc': state_dict, 'head': ..., 'tags': [...]}"""
    m = load_clsril(train=False)
    ck = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(ck["enc"])
    return m.to(device).eval()


@torch.no_grad()
def frame_features(model, wav, device="cuda:0", dtype=torch.bfloat16):
    """wav: float32 numpy [N] at 16 kHz (raw, not normalised -- CLSRIL-23 uses normalize=False). Returns [T, 768] float32 numpy, 20 ms/frame."""
    x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=dtype):
        h = model(x).last_hidden_state
    return h[0].float().cpu().numpy()
