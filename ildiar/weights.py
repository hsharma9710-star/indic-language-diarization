"""Locate / download the model weights (one file, about 430 MB)."""
import os
import urllib.request

DEFAULT_NAME = "diarizer.pt"
# Set this to the GitHub release asset URL of your copy of the repository, or export ILDIAR_WEIGHTS_URL.
DEFAULT_URL = os.environ.get("ILDIAR_WEIGHTS_URL", "")


def find_weights(path=None):
    cands = [path, os.environ.get("ILDIAR_WEIGHTS"), os.path.join("weights", DEFAULT_NAME),
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", DEFAULT_NAME)]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    if DEFAULT_URL:
        dst = os.path.join("weights", DEFAULT_NAME)
        os.makedirs("weights", exist_ok=True)
        print(f"downloading weights from {DEFAULT_URL} ...")
        urllib.request.urlretrieve(DEFAULT_URL, dst)
        return dst
    raise FileNotFoundError(
        "weights file not found. Download diarizer.pt from the repository's GitHub release, put it in ./weights/, "
        "or set ILDIAR_WEIGHTS=/path/to/diarizer.pt (see README).")
