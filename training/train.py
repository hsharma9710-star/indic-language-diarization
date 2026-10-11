import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Train W2V-ES / W2V-EA: joint loss lambda*CE(transformer) + (1-lambda)*CE(aux), class-weighted,
Adam 1e-4, cosine annealing, 60 epochs, batch 32 (same recipe as the earlier correctjagbandhu study).

python train.py --pooling attn --feat_dir_train .. --feat_dir_dev .. --tsv_train .. --tsv_dev .. --savedir ..
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ildiar.classes import SYMS
from dataset import LangDiarDataset, collate_fn
from model import W2V_ES, W2V_EA, get_atten_mask


def run_epoch(model, loader, crit, opt, lam, dev, training):
    model.train() if training else model.eval()
    tl = tc = tb = 0
    with torch.set_grad_enabled(training):
        for feats, labels, seq_len, _ in loader:
            feats, labels = feats.to(dev), labels.to(dev)
            mask = get_atten_mask(seq_len, feats.size(0), device=dev)
            tr, aux = model(feats, seq_len, mask)
            n = tr.shape[-1]
            tgt = labels.reshape(-1)
            loss = lam * crit(tr.reshape(-1, n), tgt) + (1 - lam) * crit(aux.reshape(-1, n), tgt)
            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            v = tgt != -100
            tc += (tr.reshape(-1, n).argmax(-1)[v] == tgt[v]).sum().item()
            tb += v.sum().item()
            tl += loss.item()
    return tl / max(len(loader), 1), tc / max(tb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", required=True, choices=["stat", "attn"])
    ap.add_argument("--feat_dir_train", required=True)
    ap.add_argument("--feat_dir_dev", required=True)
    ap.add_argument("--tsv_train", required=True)
    ap.add_argument("--tsv_dev", required=True)
    ap.add_argument("--savedir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lmbda", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    os.makedirs(a.savedir, exist_ok=True)
    dev = torch.device("cuda:0")
    tr, dv = LangDiarDataset(a.tsv_train, a.feat_dir_train), LangDiarDataset(a.tsv_dev, a.feat_dir_dev)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                    collate_fn=collate_fn, drop_last=True, persistent_workers=True)
    dl = DataLoader(dv, batch_size=a.batch, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    M = W2V_ES if a.pooling == "stat" else W2V_EA
    model = M(n_lang=len(SYMS), device=dev).to(dev)

    cnt = np.zeros(len(SYMS))
    for _, s in tr.entries:
        for c in set(s):
            cnt[tr.char_to_id[c]] += s.count(c)
    w = torch.tensor(cnt.sum() / (len(SYMS) * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    print("class weights", dict(zip(SYMS, np.round(w.cpu().numpy(), 3))), flush=True)
    crit = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, hist = 1e9, []
    for ep in range(1, a.epochs + 1):
        l1, a1 = run_epoch(model, tl, crit, opt, a.lmbda, dev, True)
        l2, a2 = run_epoch(model, dl, crit, opt, a.lmbda, dev, False)
        sch.step()
        print(f"[{a.pooling}] epoch {ep:03d}/{a.epochs} train_loss={l1:.4f} train_acc={a1:.4f} "
              f"dev_loss={l2:.4f} dev_acc={a2:.4f}", flush=True)
        hist.append(dict(epoch=ep, train_loss=l1, train_acc=a1, dev_loss=l2, dev_acc=a2))
        pd.DataFrame(hist).to_csv(f"{a.savedir}/history.csv", index=False)
        ck = dict(epoch=ep, model_state_dict=model.state_dict(), dev_loss=l2, dev_acc=a2, args=dict(vars(a), tags=SYMS))
        if l2 < best:
            best = l2
            torch.save(ck, f"{a.savedir}/best.pt")
            print("  -> best", flush=True)
    torch.save(ck, f"{a.savedir}/last.pt")


if __name__ == "__main__":
    main()
