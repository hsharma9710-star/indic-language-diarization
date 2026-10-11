"""Pack the fine-tuned encoder and the trained diarization head into the single `diarizer.pt` used by the tool.

python scripts/build_weight_pack.py --encoder stage2_best.pt --head e2e_best.pt --config-dir clsril23_hf \
       --pooling stat --out weights/diarizer.pt
"""
import argparse

import torch
from transformers import Wav2Vec2Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, help="checkpoint with key 'enc' (fine-tuned CLSRIL-23 encoder state dict)")
    ap.add_argument("--head", required=True, help="checkpoint with key 'model_state_dict' (W2V-ES / W2V-EA head)")
    ap.add_argument("--config-dir", required=True, help="directory holding the CLSRIL-23 config.json")
    ap.add_argument("--pooling", choices=["stat", "attn"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    cfg = Wav2Vec2Config.from_pretrained(a.config_dir)
    cfg.layerdrop = 0.0
    cfg.apply_spec_augment = False
    cfg.mask_time_prob = 0.0
    enc = torch.load(a.encoder, map_location="cpu")["enc"]
    enc = {k: v for k, v in enc.items() if k != "masked_spec_embed"}
    head = torch.load(a.head, map_location="cpu")["model_state_dict"]
    pack = dict(encoder_config=cfg.to_dict(), encoder=enc, head=head, pooling=a.pooling,
                meta=dict(note=a.note, classes="S E as bn gu hi kn ml mr ne pa sa ta te ur"))
    torch.save(pack, a.out)
    n_enc = sum(v.numel() for v in enc.values())
    n_head = sum(v.numel() for v in head.values())
    print(f"saved {a.out}: encoder {n_enc / 1e6:.1f} M params, head {n_head / 1e6:.1f} M params, pooling={a.pooling}")


if __name__ == "__main__":
    main()
