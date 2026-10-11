import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Evaluate a W2V-E2E checkpoint on a feature directory + label TSV (200 ms bins).

Two scorings, both reported:
  strict       : full 15-way argmax; an Indian-language prediction that is not the recording's own language counts as an error
  known-pair   : logits restricted to {S, English, the recording's Indian language} (the paper's 3-class setting)
Metrics: bin accuracy, per-class Jaccard, JER = 1 - mean Jaccard over (S, native, English), DER =
(ref-speech bins with wrong/missed class + ref-silence bins labelled as speech) / ref-speech bins, confusion matrix.

python evaluate.py --checkpoint best.pt --pooling attn --feat_dir F --tsv labels.tsv --native M --out_json out.json [--per_recording]
(--native: the tag of the recording pair's Indian language, e.g. M for Malayalam, K for Kannada; for the synthetic
 multilingual dev set use --native any, which maps every Indian language to 'native' in both reference and hypothesis.)
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ildiar.classes import SYMS
from dataset import LangDiarDataset, collate_fn
from model import W2V_ES, W2V_EA, get_atten_mask

S_ID, E_ID = SYMS.index("S"), SYMS.index("E")


@torch.no_grad()
def predict(model, loader, dev):
    out = []
    model.eval()
    for feats, labels, seq_len, ids in loader:
        feats = feats.to(dev)
        mask = get_atten_mask(seq_len, feats.size(0), device=dev)
        logits, _ = model(feats, seq_len, mask)
        for i, L in enumerate(seq_len):
            out.append((ids[i], labels[i, :L].numpy(), logits[i, :L].float().cpu().numpy()))
    return out


def to3(ids, native_ids):
    """collapse to 0=S 1=native 2=English 3=other-Indian"""
    o = np.full(ids.shape, 3)
    o[ids == S_ID] = 0
    o[ids == E_ID] = 2
    for n in native_ids:
        o[ids == n] = 1
    return o


def scores(ref3, hyp3):
    ref_sp = ref3 != 0
    der = (((hyp3 != ref3) & ref_sp).sum() + ((ref3 == 0) & (hyp3 != 0)).sum()) / max(ref_sp.sum(), 1)
    jac = []
    for c in range(3):
        r, h = ref3 == c, hyp3 == c
        u = np.logical_or(r, h).sum()
        jac.append(float(np.logical_and(r, h).sum() / u) if u else 1.0)
    return dict(accuracy=float((ref3 == hyp3).mean()), jer=float(1 - np.mean(jac)), der=float(der),
                jaccard=dict(zip(["S", "native", "English"], jac)), n_bins=int(len(ref3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pooling", required=True, choices=["stat", "attn"])
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--native", required=True, help="tag of the pair's Indian language (M, K, ...) or 'any'")
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--per_recording", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    ds = LangDiarDataset(a.tsv, a.feat_dir)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)
    ck = torch.load(a.checkpoint, map_location=dev)
    M = W2V_ES if a.pooling == "stat" else W2V_EA
    model = M(n_lang=len(SYMS), device=dev).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    res = predict(model, dl, dev)

    all_ids = list(range(2, len(SYMS)))
    native_ids = all_ids if a.native == "any" else [SYMS.index(a.native)]
    groups = defaultdict(list)
    for u, ref, lg in res:
        groups["ALL"].append((ref, lg))
        if a.per_recording:
            groups[re.sub(r"_c\d+$", "", u)].append((ref, lg))

    out = {}
    for g, items in groups.items():
        ref = np.concatenate([r for r, _ in items])
        lg = np.concatenate([l for _, l in items])
        hyp_strict = lg.argmax(-1)
        keep = [S_ID, E_ID] + native_ids
        sub = lg[:, keep]
        hyp_known = np.array(keep)[sub.argmax(-1)]
        ref3 = to3(ref, native_ids)
        o = dict(strict=scores(ref3, to3(hyp_strict, native_ids)), known_pair=scores(ref3, to3(hyp_known, native_ids)))
        if g == "ALL":
            cm = np.zeros((len(SYMS), len(SYMS)), dtype=np.int64)
            np.add.at(cm, (ref, hyp_strict), 1)
            o["confusion_15x15"] = cm.tolist()
            cm3 = np.zeros((3, 4), dtype=np.int64)
            np.add.at(cm3, (ref3, to3(hyp_strict, native_ids)), 1)
            o["confusion_strict_rows_ref_SNE_cols_SNEO"] = cm3.tolist()
        out[g] = o
    for g, o in out.items():
        s, k = o["strict"], o["known_pair"]
        print(f"{g:10s} strict: acc {s['accuracy']*100:.1f} JER {s['jer']*100:.1f} DER {s['der']*100:.1f} | "
              f"known-pair: acc {k['accuracy']*100:.1f} JER {k['jer']*100:.1f} DER {k['der']*100:.1f}  (bins {s['n_bins']})")
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
