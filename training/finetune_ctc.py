import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Stage 2 with the paper's CTC objective on language-tag sequences (CLSRIL-23 + Linear(768, 15+blank)).

Target modes
  collapsed : the order of language tags inside the 20 s window, consecutive duplicates merged   (S E K S E ...)
  perbin    : one token per 200 ms bin                                                        (S S K K K K E ...)
Recipe (chosen to leave the blank plateau): 1500 head-only warm-up steps (encoder frozen, head lr 1e-3), then encoder lr 3e-5
with linear decay, CNN frozen, SpecAugment time masking p=0.05, bf16 autocast, CTC computed in fp32, 10 epochs.

python finetune_ctc.py <data name> <out dir> <epochs> <collapsed|perbin>
"""
import os, random, sys, time
import numpy as np, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from ildiar.classes import SYMS
from hf_utils import ROOT, load_clsril

NAME, OUT, EPOCHS, MODE = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
SYN = f"{ROOT}/synthetic_data/{NAME}"
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda:0")
torch.manual_seed(0); random.seed(0); np.random.seed(0)
CID = {c: i for i, c in enumerate(SYMS)}
NC = len(SYMS) + 1                       # blank = 0, tag i -> i + 1
WARM = 1500


def load(split):
    d = {}
    for l in open(f"{SYN}/{split}/labels.tsv"):
        n, s = l.rstrip("\n").split("\t")
        d[n] = np.array([CID[c] for c in s])
    return d


def target(lab):
    if MODE == "collapsed":
        keep = np.concatenate([[True], lab[1:] != lab[:-1]])
        lab = lab[keep]
    return lab + 1


def levenshtein(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


tr, dv = load("train"), load("dev")
enc = load_clsril(train=True).to(dev)
for p in enc.feature_extractor.parameters():
    p.requires_grad = False
head = nn.Linear(768, NC).to(dev)
enc_params = [p for p in enc.parameters() if p.requires_grad]
opt = torch.optim.AdamW([{"params": enc_params, "lr": 3e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
WB, BS = 100, 12
names = sorted(tr)
spe = len(names) // BS
total = EPOCHS * spe


def wav(split, n):
    return sf.read(f"{SYN}/{split}/wav/{n}.wav", dtype="float32")[0]


def decode(lp):                           # greedy CTC
    ids = lp.argmax(-1)
    out, prev = [], 0
    for t in ids.tolist():
        if t != prev and t != 0:
            out.append(t)
        prev = t
    return out


def evaluate():
    enc.eval(); head.eval()
    errs = ref_len = blank_frac = 0
    n = 0
    with torch.no_grad():
        for u in sorted(dv)[::6]:
            x = torch.from_numpy(wav("dev", u)[: 16000 * 20]).float().unsqueeze(0).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(enc(x).last_hidden_state)
            lp = lg[0].float().log_softmax(-1)
            hyp = decode(lp)
            ref = target(dv[u][:100]).tolist()
            if MODE == "perbin":
                pass
            errs += levenshtein(hyp, ref); ref_len += len(ref); n += 1
            blank_frac += (lp.argmax(-1) == 0).float().mean().item()
    return errs / max(ref_len, 1), blank_frac / max(n, 1)


step = 0
best = 9.9
for ep in range(1, EPOCHS + 1):
    enc.train(); head.train(); enc.feature_extractor.eval()
    random.shuffle(names); tl = 0.0; k = 0; t0 = time.time()
    for i in range(0, len(names) - BS + 1, BS):
        xs, tg = [], []
        for n in names[i:i + BS]:
            lab = dv_lab = tr[n]
            nb = len(lab)
            if nb <= WB:
                continue
            b0 = random.randint(0, nb - WB)
            x = wav("train", n)
            xs.append(x[b0 * 3200:(b0 + WB) * 3200]); tg.append(target(lab[b0:b0 + WB]))
        if len(xs) < 2:
            continue
        x = torch.from_numpy(np.stack(xs)).float().to(dev)
        warm = step < WARM
        for g, base in zip(opt.param_groups, (3e-5, 1e-3)):
            if g is opt.param_groups[0]:
                g["lr"] = 0.0 if warm else base * max(0.05, 1 - (step - WARM) / max(total - WARM, 1))
            else:
                g["lr"] = base * min(1.0, (step + 1) / 200) * (1.0 if warm else max(0.1, 1 - (step - WARM) / max(total - WARM, 1)))
        if warm:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = enc(x).last_hidden_state
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(h.detach())
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = head(enc(x).last_hidden_state)
        lp = lg.float().log_softmax(-1).transpose(0, 1)                    # [T, B, C]
        tl_ = torch.tensor([len(t) for t in tg], dtype=torch.long)
        loss = F.ctc_loss(lp, torch.from_numpy(np.concatenate(tg)).long(), torch.full((lp.shape[1],), lp.shape[0], dtype=torch.long), tl_,
                          blank=0, reduction="sum", zero_infinity=True) / len(tg)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 5.0)
        opt.step(); step += 1
        tl += loss.item(); k += 1
        if k % 250 == 0:
            print(f"  ep {ep} step {k} ctc {tl / k:.2f} ({time.time() - t0:.0f}s)", flush=True)
    ler, bf = evaluate()
    print(f"epoch {ep} train_ctc {tl / max(k, 1):.3f} dev_LER {ler:.3f} blank_frac {bf:.2f} time {time.time() - t0:.0f}s", flush=True)
    if ler < best:
        best = ler
        torch.save({"enc": enc.state_dict(), "head": head.state_dict(), "tags": SYMS, "epoch": ep, "dev_ler": float(ler), "mode": MODE}, f"{OUT}/best.pt")
print("done best dev LER", best)
