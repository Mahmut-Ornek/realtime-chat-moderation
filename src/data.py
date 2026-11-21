"""Loads the OffensEval-2019 splits (train / validation / test) from the Hugging Face Hub."""
from datasets import load_dataset

from .config import DATASET_CONFIG, DATASET_NAME


def load_splits():
    """Returns {"train": (texts, labels), "validation": (...), "test": (...)}.

    Label 0 = clean (non-offensive), 1 = offensive. The test split is the official
    OffensEval-2019 test set and is used for every model in the results table.
    """
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    return {split: (list(ds[split]["text"]), list(ds[split]["label"])) for split in ("train", "validation", "test")}
