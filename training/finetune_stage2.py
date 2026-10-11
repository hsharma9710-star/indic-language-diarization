import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Stage 2 (language fine-tuning of CLSRIL-23) on the synthetic multilingual data.

The paper fine-tunes with CTC; in the earlier study CTC did not leave its blank plateau, so here every 20 ms frame is classified
with cross-entropy against the tag of its 200 ms bin (15 tags: S, E and one per Indian language). CNN feature encoder frozen,
transformer + a temporary Linear(768, 15) head trained. The head is discarded afterwards; only the encoder is used for features.

python finetune_stage2.py <synthetic data name> <out dir> <epochs>
"""
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from ildiar.classes import SYMS
from hf_utils import ROOT, load_clsril

NAME, OUT, EPOCHS = sys.argv[1], sys.argv[2], int(sys.argv[3])
SYN = f"{ROOT}/synthetic_data/{NAME}"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda:0")
torch.manual_seed(0); random.seed(0); np.random.seed(0)
CID = {c: i for i, c in enumerate(SYMS)}


def load(split):
    d = {}
    for l in open(f"{SYN}/{split}/labels.tsv"):
        n, s = l.rstrip("\n").split("\t")
        d[n] = np.array([CID[c] for c in s])
    return d


tr, dv = load("train"), load("dev")
enc = load_clsril(train=True).to(dev)
for p in enc.feature_extractor.parameters():
    p.requires_grad = False
head = nn.Linear(768, len(SYMS)).to(dev)
opt = torch.optim.AdamW([{"params": [p for p in enc.parameters() if p.requires_grad], "lr": 5e-5},
                         {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
WB, BS = 100, 12                       # 20 s windows, 12 per step (240 s)
names = sorted(tr)
total = EPOCHS * (len(names) // BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 300) * max(0.05, 1 - s / total))
cnt = np.bincount(np.concatenate(list(tr.values())), minlength=len(SYMS))
wts = torch.tensor(cnt.sum() / (len(SYMS) * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
print("tags", SYMS, "counts", cnt.tolist(), flush=True)


def wav(split, n):
    return sf.read(f"{SYN}/{split}/wav/{n}.wav", dtype="float32")[0]


def evaluate():
    enc.eval(); head.eval()
    yt, yp = [], []
    with torch.no_grad():
        for n in sorted(dv)[::4]:
            x = torch.from_numpy(wav("dev", n)).float().unsqueeze(0).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(enc(x).last_hidden_state)
            nb = len(dv[n]); T = lg.shape[1] // 10 * 10
            fr = lg[0, :T].float().view(-1, 10, len(SYMS)).mean(1)
            m = min(len(fr), nb)
            yp.append(fr[:m].argmax(-1).cpu().numpy()); yt.append(dv[n][:m])
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return (yt == yp).mean()


best = 0.0
for ep in range(1, EPOCHS + 1):
    enc.train(); head.train(); enc.feature_extractor.eval()
    random.shuffle(names); tl = 0.0; k = 0; t0 = time.time()
    for i in range(0, len(names) - BS + 1, BS):
        xs, ys = [], []
        for n in names[i:i + BS]:
            lab = tr[n]; nb = len(lab)
            if nb <= WB:
                continue
            b0 = random.randint(0, nb - WB)
            x = wav("train", n)
            xs.append(x[b0 * 3200:(b0 + WB) * 3200]); ys.append(np.repeat(lab[b0:b0 + WB], 10))
        if len(xs) < 2:
            continue
        x = torch.from_numpy(np.stack(xs)).float().to(dev); y = torch.from_numpy(np.stack(ys)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = head(enc(x).last_hidden_state)
        T = min(lg.shape[1], y.shape[1])
        loss = F.cross_entropy(lg[:, :T].float().reshape(-1, len(SYMS)), y[:, :T].reshape(-1), weight=wts)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0); opt.step(); sched.step()
        tl += loss.item(); k += 1
        if k % 200 == 0:
            print(f"  ep {ep} step {k} loss {tl / k:.4f} ({time.time() - t0:.0f}s)", flush=True)
    acc = evaluate()
    print(f"epoch {ep} train_loss {tl / max(k, 1):.4f} dev_bin_acc {acc:.4f} time {time.time() - t0:.0f}s", flush=True)
    if acc > best:
        best = acc
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "tags": SYMS, "epoch": ep, "dev_acc": float(acc)}, f"{OUT}/best.pt")
print("done best dev acc", best)
