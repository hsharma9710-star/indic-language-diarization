# Training and evaluation scripts

These are the scripts that produced the model in `weights/diarizer.pt`. They are provided for reproducibility; **you do not need
them to use the tool**. Set `ILDIAR_ROOT` to a working directory and arrange it as below.

```
$ILDIAR_ROOT/
  models/clsril23_hf/          config.json + pytorch_model.bin of Harveenchadha/wav2vec2-pretrained-clsril-23-10k
  voxlingua/raw_<lang>/<lang>/*.wav     the unzipped VoxLingua107 archives (en, as, bn, gu, hi, kn, ml, mr, ne, pa, sa, ta, te, ur)
  synthetic_data/<name>/{train,dev}/{wav,labels.tsv,turns.jsonl}      (created by make_synth_multi.py)
```

Order (run from the repository root):

| step | script | what it does |
|---|---|---|
| 1 | `training/make_synth_multi.py <name> as,bn,gu,hi,kn,ml,mr,ne,pa,sa,ta,te,ur 100 6 1` | builds synthetic utterances: English + 1 or 2 Indian languages, 20-40 s, VAD-derived silence, one noise floor per utterance; train/dev split by YouTube video id |
| 2a | `training/finetune_stage2.py <name> <out> 10` | Stage 2 with frame-level cross-entropy (the model shipped in this repository) |
| 2b | `training/finetune_ctc.py <name> <out> 10 perbin` | Stage 2 with the paper's CTC objective (`collapsed` = run-length-collapsed target) |
| 3 | `training/extract_features.py --enc <ckpt or pre> --wav_dir ... --out_dir ...` | frozen 768-d / 20 ms features (float16 .npy) |
| 4 | `training/train.py --pooling stat\|attn ...` | trains the pooling + self-attention head (60 epochs, Adam 1e-4, cosine, batch 32, class-weighted joint loss 0.5/0.5) |
| 5 | `training/evaluate.py`, `training/cluster_eval.py` | bin-level accuracy / JER / DER, strict and known-pair; clustering of the 256-d embeddings |
| 6 | `scripts/build_weight_pack.py` | packs encoder + head into `diarizer.pt` |

Hyper-parameters, the data statistics and every result (24 models: 4 front-ends x 2 poolings x 3 seeds) are in `docs/RESULTS.md`.

`classes.py` here re-exports `ildiar/classes.py`; `model.py` is the training-time version of the head (with the auxiliary head and
attention-mask support), whose weights load into `ildiar/model.py`.
