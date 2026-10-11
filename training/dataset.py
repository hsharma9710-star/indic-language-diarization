import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from torch.utils.data import Dataset

from ildiar.classes import SYMS

FRAMES_PER_POOL = 10


class LangDiarDataset(Dataset):
    """TSV `utt<TAB>label_string` (one symbol per 200 ms bin, symbols from classes.SYMS)
    + pre-extracted [T, 768] float16 features (.npy, ~20 ms/frame)."""

    def __init__(self, tsv_path, feat_dir):
        self.feat_dir = feat_dir
        self.char_to_id = {c: i for i, c in enumerate(SYMS)}
        self.entries = []
        for line in open(tsv_path, encoding="utf-8"):
            line = line.rstrip("\n")
            if line.strip():
                u, s = line.split("\t")
                self.entries.append((u, s))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        u, s = self.entries[i]
        feat = np.load(f"{self.feat_dir}/{u}.npy").astype(np.float32)
        lab = np.array([self.char_to_id[c] for c in s], dtype=np.int64)
        nb = max(min(feat.shape[0] // FRAMES_PER_POOL, len(lab)), 1)
        return torch.from_numpy(feat[: nb * FRAMES_PER_POOL]), torch.from_numpy(lab[:nb]), u


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)
    feats, labels, ids = zip(*batch)
    lens = [l.shape[0] for l in labels]
    mb = max(lens)
    pf = torch.zeros(len(batch), mb * FRAMES_PER_POOL, feats[0].shape[1])
    pl = torch.full((len(batch), mb), -100, dtype=torch.long)
    for i, (f, l) in enumerate(zip(feats, labels)):
        pf[i, : f.shape[0]] = f
        pl[i, : l.shape[0]] = l
    return pf, pl, lens, ids
