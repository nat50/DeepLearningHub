"""Yahoo Answers text classification dataset and dataloader utilities.

Mirrors the structure of ``assignment1/image_classification/dataset.py``:
- ``YahooAnswersDataset``  – core PyTorch Dataset
- ``LSTMDataset`` / ``TransformerDataset`` – model-specific wrappers
- ``create_dataloaders()`` – one-call factory returning everything the
  training pipeline needs
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

from data_download import download_yahoo_answers

# ── constants ────────────────────────────────────────────────────────────────

CLASS_NAMES: Dict[int, str] = {
    0: "Society & Culture",
    1: "Science & Mathematics",
    2: "Health",
    3: "Education & Reference",
    4: "Computers & Internet",
    5: "Sports",
    6: "Business & Finance",
    7: "Entertainment & Music",
    8: "Family & Relationships",
    9: "Politics & Government",
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
DEFAULT_VOCAB_SIZE = 50_000

STOP_WORDS = frozenset({
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone",
    "anything", "anyway", "anywhere", "are", "aren", "arent", "around", "as",
    "at", "back", "be", "became", "because", "become", "becomes", "becoming",
    "been", "before", "beforehand", "behind", "being", "below", "beside",
    "besides", "between", "beyond", "both", "but", "by", "can", "cannot",
    "cant", "co", "could", "couldn", "couldnt", "d", "did", "didn", "didnt",
    "do", "does", "doesn", "doesnt", "doing", "don", "dont", "down", "during",
    "each", "either", "else", "elsewhere", "enough", "etc", "even", "ever",
    "every", "everyone", "everything", "everywhere", "few", "for", "from",
    "further", "get", "gets", "getting", "go", "goes", "going", "gone", "got",
    "had", "hadn", "hadnt", "has", "hasn", "hasnt", "have", "haven", "havent",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "however", "i", "if", "in", "into", "is", "isn", "isnt", "it",
    "its", "itself", "just", "know", "ll", "m", "made", "many", "may", "me",
    "might", "more", "moreover", "most", "much", "must", "my", "myself",
    "neither", "never", "nevertheless", "no", "nobody", "none", "noone", "nor",
    "not", "nothing", "now", "nowhere", "o", "of", "off", "often", "on",
    "once", "one", "only", "or", "other", "others", "otherwise", "our", "ours",
    "ourselves", "out", "over", "own", "per", "perhaps", "please", "re", "s",
    "same", "she", "should", "shouldn", "shouldnt", "so", "some", "somehow",
    "someone", "something", "sometime", "sometimes", "somewhere", "still",
    "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "therefore", "these", "they", "this", "those", "though",
    "through", "throughout", "to", "together", "too", "toward", "towards",
    "under", "until", "up", "upon", "us", "ve", "very", "was", "wasn", "wasnt",
    "we", "well", "were", "weren", "werent", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "who",
    "whoever", "whole", "whom", "whose", "why", "will", "with", "within",
    "without", "won", "wont", "would", "wouldn", "wouldnt", "yet", "you",
    "your", "yours", "yourself", "yourselves",
})


# ── text preprocessing ───────────────────────────────────────────────────────

def clean_text(text: Any) -> str:
    """Lowercase, remove URLs / HTML / punctuation, normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace literal escape sequences from CSV parsing
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = text.replace("\n", " ").replace("\r", " ")
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── vocabulary ───────────────────────────────────────────────────────────────

def build_vocab(
    texts: Sequence[str],
    max_vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build word→idx and idx→word mappings from a corpus of cleaned texts."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())

    # Reserve indices 0 and 1 for PAD and UNK
    word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _ in counter.most_common(max_vocab_size):
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word


def encode_text(text: str, word2idx: Dict[str, int]) -> List[int]:
    """Encode a cleaned string into a list of integer indices."""
    return [word2idx.get(w, UNK_IDX) for w in text.split()]


# ── core dataset ─────────────────────────────────────────────────────────────

class YahooAnswersDataset(Dataset):
    """Yahoo Answers topic classification dataset (10 classes, 1.4M samples)."""

    def __init__(self, csv_path: str | Path | None = None):
        if csv_path is None:
            csv_path = download_yahoo_answers()
        csv_path = Path(csv_path)

        column_names = ["class_index", "question_title", "question_content", "best_answer"]
        df = pd.read_csv(csv_path, header=None, names=column_names)

        # Combine text fields
        df["full_text"] = (
            df["question_title"].fillna("")
            + " "
            + df["question_content"].fillna("")
            + " "
            + df["best_answer"].fillna("")
        )

        # Shift labels 1-10 → 0-9
        df["label"] = df["class_index"] - 1

        # Clean text
        df["cleaned_text"] = df["full_text"].apply(clean_text)

        self.texts: List[str] = df["cleaned_text"].tolist()
        self.labels: List[int] = df["label"].tolist()
        self.num_classes = len(CLASS_NAMES)
        self.class_names = [CLASS_NAMES[i] for i in range(self.num_classes)]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]


# ── model-specific dataset wrappers ──────────────────────────────────────────

class LSTMDataset(Dataset):
    """Wraps cleaned texts as integer-encoded, truncated sequences for LSTM."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        word2idx: Dict[str, int],
        max_seq_len: int = 256,
    ):
        self.encoded = [encode_text(t, word2idx)[:max_seq_len] for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        seq = self.encoded[idx] if self.encoded[idx] else [UNK_IDX]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class TransformerDataset(Dataset):
    """Wraps cleaned texts for HuggingFace tokenizer-based models."""

    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.texts[idx], int(self.labels[idx])


# ── collate functions ────────────────────────────────────────────────────────

def lstm_collate_fn(batch):
    """Pad variable-length sequences for LSTM."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return padded, lengths, labels


def make_transformer_collate_fn(tokenizer, max_length: int = 256):
    """Return a collate function bound to a specific tokenizer."""

    def collate_fn(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return encoded["input_ids"], encoded["attention_mask"], labels

    return collate_fn


# ── splits & dataloaders ─────────────────────────────────────────────────────

def make_stratified_split_indices(
    labels: Sequence[int],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Deterministic stratified train/val/test index split."""
    indices = np.arange(len(labels))
    labels_arr = np.asarray(labels)

    train_idx, temp_idx = train_test_split(
        indices, train_size=train_size, random_state=seed, stratify=labels_arr,
    )
    val_ratio = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_ratio, random_state=seed, stratify=labels_arr[temp_idx],
    )
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def create_dataloaders(
    model_type: str = "lstm",
    batch_size: int = 64,
    num_workers: int = 2,
    max_seq_len: int = 256,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    tokenizer=None,
    csv_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Build train/val/test dataloaders for text classification.

    Parameters
    ----------
    model_type : ``"lstm"`` or ``"transformer"``
    tokenizer : HuggingFace tokenizer (required when ``model_type="transformer"``)
    """
    base_ds = YahooAnswersDataset(csv_path=csv_path)
    split_indices = make_stratified_split_indices(
        base_ds.labels, train_size=train_size, val_size=val_size,
        test_size=test_size, seed=seed,
    )

    train_texts = [base_ds.texts[i] for i in split_indices["train"]]
    train_labels = [base_ds.labels[i] for i in split_indices["train"]]
    val_texts = [base_ds.texts[i] for i in split_indices["val"]]
    val_labels = [base_ds.labels[i] for i in split_indices["val"]]
    test_texts = [base_ds.texts[i] for i in split_indices["test"]]
    test_labels = [base_ds.labels[i] for i in split_indices["test"]]

    extra: Dict[str, Any] = {}

    if model_type == "lstm":
        word2idx, idx2word = build_vocab(train_texts, max_vocab_size=vocab_size)
        extra["word2idx"] = word2idx
        extra["idx2word"] = idx2word
        extra["vocab_size"] = len(word2idx)

        train_ds = LSTMDataset(train_texts, train_labels, word2idx, max_seq_len)
        val_ds = LSTMDataset(val_texts, val_labels, word2idx, max_seq_len)
        test_ds = LSTMDataset(test_texts, test_labels, word2idx, max_seq_len)
        collate = lstm_collate_fn

    elif model_type == "transformer":
        if tokenizer is None:
            raise ValueError("tokenizer is required for model_type='transformer'")
        train_ds = TransformerDataset(train_texts, train_labels)
        val_ds = TransformerDataset(val_texts, val_labels)
        test_ds = TransformerDataset(test_texts, test_labels)
        collate = make_transformer_collate_fn(tokenizer, max_length=max_seq_len)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": base_ds.class_names,
        "num_classes": base_ds.num_classes,
        "split_sizes": {k: len(v) for k, v in split_indices.items()},
        "model_type": model_type,
        **extra,
    }


if __name__ == "__main__":
    loaders = create_dataloaders(model_type="lstm", batch_size=8, num_workers=0)
    print("Num classes:", loaders["num_classes"])
    print("Vocab size:", loaders["vocab_size"])
    print("Split sizes:", loaders["split_sizes"])
    padded, lengths, labels = next(iter(loaders["train_loader"]))
    print("LSTM batch shapes:", tuple(padded.shape), tuple(lengths.shape), tuple(labels.shape))
