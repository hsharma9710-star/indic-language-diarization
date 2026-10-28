# English second opinion for the diarizer

Problem: on real Hindi + English news audio the diarizer labelled everything Hindi (English probability about 1 %).
Fix: Whisper-tiny's language detector runs on 4 s windows (one every 2 s). A speech bin that the diarizer calls an Indian language is
switched to English where the interpolated English probability is at least 0.3.

    python english_boost/english_boost.py talk.mp3 -o out --languages hi

Outputs `*_before.rttm / .png` and `*_after.rttm / .png`, plus pooled audio of the corrected result. The first run downloads
`openai/whisper-tiny` (about 150 MB) from Hugging Face. The website has the same check built in (an ONNX copy, on by default).

## Results

| Test | Before | After |
|---|---|---|
| 6 held-out real recordings (English + Malayalam / Kannada, 4-minute excerpts), DER | 24.7 % | 16.4 % |
| same 12 recordings, 6 used to choose the threshold, DER | 24.8 % | 17.7 % |
| Hindi news clip (30 s): English found | 0 % | about 65 %, no English wrongly placed in the Hindi parts |

The window length, hop and threshold were chosen on 6 of the 12 recordings and checked on the other 6.

## Limits
- The Hindi news result is one 30 s clip and approximate; English at the very start of that clip is still labelled Hindi.
- The 12 recordings are 4-minute excerpts (minutes 10-14) of the 27, not the full set; the published 25.7 % figure is unchanged.
- The 4 s window means English stretches shorter than about 2 s can still be missed.
