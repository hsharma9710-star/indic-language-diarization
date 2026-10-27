# English second opinion for the diarizer

Problem: on real Hindi + English news audio the diarizer labelled everything Hindi (English probability about 1 %).
Fix: run an English speech recogniser (`facebook/wav2vec2-base-960h`) next to the diarizer. A 200 ms bin that the diarizer calls
an Indian language is switched to English when at least 25 % of the surrounding ~4 s is made of recognised English words
(words of 4+ letters from an English word list, plus common short words).

    python english_boost/english_boost.py talk.mp3 -o out --languages hi

Outputs `*_before.rttm / .png` and `*_after.rttm / .png`, plus pooled audio of the corrected result. Needs `transformers` and `torch` (already required by the tool).
The first run downloads the recogniser (about 360 MB). The website is not affected (the recogniser is not in the browser).

## Results

| Test | Before | After |
|---|---|---|
| Delhi news clip (30 s), English regions found | 0 % | 50 % (no English wrongly placed in the Hindi regions) |
| 12 real recordings (English + Malayalam / Kannada), 4-minute excerpts, **6 held-out recordings** | DER 24.7 % | DER 22.2 % |
| same, all 12 | DER 24.8 % | DER 21.8 % |

Settings (word length 4, 4 s smoothing, threshold 0.25) were chosen on 6 of the 12 recordings only and then checked on the other 6.

## Limits
- Delhi regions were located by reading the recogniser's transcript, so that check is approximate and one clip only.
- English spoken in a heavy accent is recognised less well, so some English is still missed (about half in the Delhi clip).
- The 12 recordings are 4-minute excerpts (minutes 10-14) of the 27, not the full set; the published 25.7 % figure is unchanged.
