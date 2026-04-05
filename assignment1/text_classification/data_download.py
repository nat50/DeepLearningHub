"""Download the Yahoo Answers topic classification dataset.

The dataset is sourced from Kaggle (soumikrakshit/yahoo-answers-dataset).
If ``kagglehub`` is installed and configured the download is automatic;
otherwise the user can place ``train.csv`` manually under ``./data/``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

TARGET_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_CSV = os.path.join(TARGET_DIR, "train.csv")


def download_yahoo_answers() -> str:
    """Download the Yahoo Answers dataset and return the path to ``train.csv``.

    Lookup order:
    1. ``data/train.csv`` already present → reuse.
    2. ``train.csv`` next to this file (notebook working-dir convention) → copy into ``data/``.
    3. Use ``kagglehub`` to download → copy ``train.csv`` into ``data/``.
    """
    os.makedirs(TARGET_DIR, exist_ok=True)

    if os.path.isfile(TRAIN_CSV):
        print("Dataset already exists!")
        print("Path to dataset file:", TRAIN_CSV)
        return TRAIN_CSV

    # Fallback: notebook may have placed train.csv in cwd
    local_csv = os.path.join(os.path.dirname(__file__), "train.csv")
    if os.path.isfile(local_csv):
        shutil.copy2(local_csv, TRAIN_CSV)
        print("Copied local train.csv →", TRAIN_CSV)
        return TRAIN_CSV

    # Download from Kaggle
    try:
        import kagglehub  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed.  Either install it (`pip install kagglehub`) "
            "or place train.csv manually under data/."
        ) from exc

    print("Downloading Yahoo Answers dataset from Kaggle …")
    kaggle_path = kagglehub.dataset_download("soumikrakshit/yahoo-answers-dataset")
    print("Kaggle download path:", kaggle_path)

    # Locate train.csv inside the Kaggle download tree
    for root, _dirs, files in os.walk(kaggle_path):
        if "train.csv" in files:
            src = os.path.join(root, "train.csv")
            shutil.copy2(src, TRAIN_CSV)
            print("Copied →", TRAIN_CSV)
            return TRAIN_CSV

    raise FileNotFoundError(
        f"train.csv not found inside Kaggle download at {kaggle_path}. "
        "Please place it manually under data/."
    )


if __name__ == "__main__":
    download_yahoo_answers()
