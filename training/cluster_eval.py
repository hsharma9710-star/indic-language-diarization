import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Clustering stage of the Mishra pipeline (variable-length language segments, no per-200 ms decision).

For every recording (all of its 30 s chunks concatenated):
  1. run the trained W2V-E2E network and keep the context-aware 256-d embedding of every 200 ms bin (+ class posteriors)
  2. speech bins = bins whose silence posterior is below 0.5
  3. L2-normalise embeddings of the speech bins and cluster them with k-means; k in {1,2,3} (English + at most 2 Indian
     languages) is chosen by silhouette score (k=1 if no k has silhouette >= --min_sil)
  4. smooth the cluster sequence with a median filter (--smooth bins) so segments are contiguous and of variable length
  5. each cluster is NAMED by the class with the highest mean posterior inside it (English or an Indian language)
Scoring: (a) named clusters with the same strict / known-pair scoring as evaluate.py; (b) diarization-style score where
clusters are mapped one-to-one to reference classes by the Hungarian algorithm (no use of the classifier's names).

python cluster_eval.py --checkpoint best.pt --pooling attn --feat_dir F --tsv labels.tsv --native M --out_json o.json
"""
import argparse
import json
import re
from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from ildiar.classes import SYMS
from dataset import LangDiarDataset, collate_fn
from evaluate import S_ID, E_ID, scores, to3
from model import W2V_ES, W2V_EA, get_atten_mask


@torch.no_grad()
def run_model(model, loader, dev):
    out = []
    model.eval()
    for feats, labels, seq_len, ids in loader:
        feats = feats.to(dev)
        mask = get_atten_mask(seq_len, feats.size(0), device=dev)
        logits, _, emb = model(feats, seq_len, mask, return_embedding=True)
        post = torch.softmax(logits.float(), -1)
        for i, L in enumerate(seq_len):
            out.append((ids[i], labels[i, :L].numpy(), post[i, :L].cpu().numpy(), emb[i, :L].float().cpu().numpy()))
    return out


def cluster_recording(post, emb, smooth, min_sil, seed=0):
    n = len(post)
    speech = post[:, S_ID] < 0.5
    hyp = np.full(n, S_ID)
    k_used = 0
    if speech.sum() < 10:
        return hyp, k_used
    x = emb[speech]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    rng = np.random.RandomState(seed)
    sub = rng.choice(len(x), min(len(x), 3000), replace=False)
    best_k, best_s, best_lab = 1, min_sil, np.zeros(len(x), dtype=int)
    for k in (2, 3):
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(x)
        s = silhouette_score(x[sub], km.labels_[sub])
        if s > best_s:
            best_k, best_s, best_lab = k, s, km.labels_
    lab = np.full(n, -1)
    lab[speech] = best_lab
    # median-filter the cluster sequence (non-speech bins keep -1 and are not smoothed over)
    sm = lab.copy()
    idx = np.where(speech)[0]
    sm_speech = median_filter(lab[idx], size=smooth, mode="nearest")
    sm[idx] = sm_speech
    # name every cluster by its mean posterior (excluding silence)
    for c in range(best_k):
        m = sm == c
        if m.any():
            p = post[m].mean(0)
            p[S_ID] = -1
            hyp[m] = int(p.argmax())
    return hyp, best_k, sm


def hungarian_score(ref3, clu, k):
    """map clusters to reference classes {S,native,English} one-to-one, then score (diarization-style)"""
    cm = np.zeros((k, 3))
    for c in range(k):
        for r in range(3):
            cm[c, r] = ((clu == c) & (ref3 == r)).sum()
    ci, ri = linear_sum_assignment(-cm)
    mp = {c: r for c, r in zip(ci, ri)}
    h = np.zeros_like(ref3)
    for c in range(k):
        h[clu == c] = mp.get(c, 3)
    h[clu == -1] = 0
    return scores(ref3, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pooling", required=True, choices=["stat", "attn"])
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--smooth", type=int, default=9, help="median filter length in 200 ms bins")
    ap.add_argument("--min_sil", type=float, default=0.05)
    ap.add_argument("--group_by", default="recording", choices=["recording", "utterance"])
    ap.add_argument("--out_json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    ds = LangDiarDataset(a.tsv, a.feat_dir)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)
    ck = torch.load(a.checkpoint, map_location=dev)
    M = W2V_ES if a.pooling == "stat" else W2V_EA
    model = M(n_lang=len(SYMS), device=dev).to(dev)
    model.load_state_dict(ck["model_state_dict"])
    res = run_model(model, dl, dev)
    native_ids = list(range(2, len(SYMS))) if a.native == "any" else [SYMS.index(a.native)]

    groups = defaultdict(list)
    for u, ref, post, emb in res:
        key = re.sub(r"_c\d+$", "", u) if a.group_by == "recording" else u
        groups[key].append((ref, post, emb))
    refs, hyps, ks, cl3, clk = [], [], [], [], []
    segs_per_hour = []
    for g, items in sorted(groups.items()):
        ref = np.concatenate([r for r, _, _ in items]); post = np.concatenate([p for _, p, _ in items]); emb = np.concatenate([e for _, _, e in items])
        out = cluster_recording(post, emb, a.smooth, a.min_sil)
        hyp, k = out[0], out[1]
        sm = out[2] if len(out) > 2 else np.full(len(ref), -1)
        refs.append(ref); hyps.append(hyp); ks.append(k); clk.append(k)
        cl3.append((to3(ref, native_ids), sm))
        nseg = 1 + int((np.diff(hyp) != 0).sum())
        segs_per_hour.append(nseg / (len(ref) * 0.2 / 3600))
    ref = np.concatenate(refs); hyp = np.concatenate(hyps)
    ref3 = to3(ref, native_ids)
    res_out = dict(named_strict=scores(ref3, to3(hyp, native_ids)),
                   k_selected=ks, mean_segments_per_hour=float(np.mean(segs_per_hour)))
    # Hungarian (name-free) score, k = max clusters used
    kmax = max(max(clk), 1)
    allr = np.concatenate([r for r, _ in cl3]); allc = np.concatenate([c for _, c in cl3])
    # clusters are per recording: offset ids so they stay distinct, then map per recording and pool
    hs = []
    off = 0
    for r3, c in cl3:
        k = int(c.max()) + 1 if (c >= 0).any() else 1
        score_h = hungarian_score(r3, c, k)
        hs.append((r3, c, k))
    pooled_ref, pooled_hyp = [], []
    for r3, c, k in hs:
        cm = np.zeros((k, 3))
        for ci in range(k):
            for ri in range(3):
                cm[ci, ri] = ((c == ci) & (r3 == ri)).sum()
        a_i, b_i = linear_sum_assignment(-cm)
        mp = {x: y for x, y in zip(a_i, b_i)}
        h = np.zeros_like(r3)
        for ci in range(k):
            h[c == ci] = mp.get(ci, 3)
        h[c == -1] = 0
        pooled_ref.append(r3); pooled_hyp.append(h)
    res_out["hungarian_name_free"] = scores(np.concatenate(pooled_ref), np.concatenate(pooled_hyp))
    s, h_ = res_out["named_strict"], res_out["hungarian_name_free"]
    print(f"clusters/recording k={ks}  segments/hour={res_out['mean_segments_per_hour']:.0f}")
    print(f"named clusters : acc {s['accuracy']*100:.1f} JER {s['jer']*100:.1f} DER {s['der']*100:.1f}")
    print(f"name-free (Hungarian): acc {h_['accuracy']*100:.1f} JER {h_['jer']*100:.1f} DER {h_['der']*100:.1f}")
    if a.out_json:
        json.dump(res_out, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
