# Results and limitations

Metric: diarization error rate on 200 ms bins, **no collar** (stricter than NIST DER's 250 ms collar):
DER = (reference-speech bins with the wrong / missed class + silence bins labelled as speech) / reference-speech bins.
Silence = time not covered by any reference segment.

## The shipped tool on real recordings

27 human-annotated recordings (`family_ld`, 15 English + Malayalam, 12 English + Kannada, 21 h) that were **never used for training
or for choosing the language-detection rule**. (The choice of cross-entropy fine-tuning + statistical pooling among the trained
variants was made from development results and, at the margin, from this test set, so the figure below is slightly optimistic.)

| mode | pooled DER |
|---|---|
| Indian language(s) named with `--languages` | **25.7 %** |
| languages detected automatically | **29.4 %** |

* The exact language set is found for 56 % of the files; in 12 of 27 files the detector adds a second, similar-sounding Indian language
  (Telugu / Kannada / Tamil / Hindi) that is not there.
* DER per recording with languages named ranges from 13 % to 42 %.

## Models trained (synthetic development set, 6.7 h; real-recording numbers in the next table)

110 h synthetic training data, 3 seeds each, 60 epochs, best checkpoint by development loss. Mean ± std.

| Stage-2 fine-tuning | Pooling | 15-class acc. | 3-class DER (dev) |
|---|---|---|---|
| CTC per-bin (paper's objective) | attention | 93.1 ± 0.2 | 3.8 ± 0.1 |
| CTC per-bin | statistical | 93.0 ± 0.3 | 4.1 ± 0.2 |
| Cross-entropy (**shipped**) | statistical | 93.0 ± 0.1 | 4.3 ± 0.1 |
| Cross-entropy | attention | 92.7 ± 0.1 | 4.5 ± 0.3 |
| CTC collapsed target | statistical | 83.0 ± 0.5 | 9.0 ± 0.4 |
| CTC collapsed target | attention | 81.6 ± 0.8 | 10.1 ± 0.9 |
| none (plain CLSRIL-23) | attention | 73.9 ± 2.1 | 10.4 ± 1.9 |
| none (plain CLSRIL-23) | statistical | 66.8 ± 0.5 | 14.1 ± 0.5 |

## The same models on the 27 real recordings (known-pair DER %, mean over 3 seeds)

| Features | Pooling | English + Malayalam | English + Kannada |
|---|---|---|---|
| CE fine-tuned | statistical | 20.3 | 32.5 |
| CE fine-tuned | attention | 21.3 | 32.1 |
| CTC per-bin | statistical | 23.1 | 32.1 |
| CTC per-bin | attention | 22.4 | 37.6 |
| CTC collapsed | statistical | 26.3 | 32.0 |
| plain CLSRIL-23 | attention | 38.1 | 48.4 |

("known-pair" = the output is restricted to silence, English and the recording's own Indian language, the setting of the original paper;
the tool does the same when `--languages` is given.) The advantage of CTC on the synthetic development set does **not** carry over
to real recordings.

## Limitations

* Trained on **synthetic** speech built from monolingual clips; every language switch is also a change of speaker and recording
  channel. Real conversation is much harder (3.8 % DER on synthetic development data vs. 26 % on real recordings).
* Real-data validation covers **English + Malayalam and English + Kannada only**. The other languages and all three-language
  mixtures are validated on synthetic data only.
* Automatic language detection is unreliable between related languages; name the languages when you can.
* Silence is poorly found on real recordings.
* Input is processed in 30 s chunks; very short switches (< 0.4 s) are often missed or mislabelled.
* Eight of CLSRIL-23's pre-training languages are not in VoxLingua107 and are not supported.
