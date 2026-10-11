"""Class set: silence, English and 13 Indian languages (the VoxLingua107 languages that CLSRIL-23 was also pre-trained on).

Class id = position in SYMS.  `SYMS` symbols are the single-character tags used in the training label files.
"""
LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "pa", "sa", "ta", "te", "ur"]
TAG = {"as": "A", "bn": "B", "gu": "G", "hi": "H", "kn": "K", "ml": "M", "mr": "R", "ne": "P",
       "pa": "J", "sa": "X", "ta": "T", "te": "L", "ur": "U"}
SYMS = ["S", "E"] + [TAG[l] for l in LANGS]

LANG_NAMES = {"en": "English", "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
              "ml": "Malayalam", "mr": "Marathi", "ne": "Nepali", "pa": "Punjabi", "sa": "Sanskrit", "ta": "Tamil",
              "te": "Telugu", "ur": "Urdu"}
SUPPORTED = ["en"] + LANGS

# class id <-> ISO 639-1 code
ID2CODE = {0: "sil", 1: "en"}
ID2CODE.update({i + 2: l for i, l in enumerate(LANGS)})
CODE2ID = {v: k for k, v in ID2CODE.items()}
SIL, ENG = 0, 1
INDIC_IDS = list(range(2, len(SYMS)))

# some corpora (e.g. the evaluation data of this project) write Kannada as KA in RTTMs
RTTM_ALIASES = {"KA": "kn"}
