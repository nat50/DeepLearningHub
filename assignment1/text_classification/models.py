"""Text classification models for the Yahoo Answers pipeline.

Mirrors ``assignment1/image_classification/models.py``:
- ``myLSTM``          – bidirectional LSTM classifier
- ``myTransformer``   – BERT-based classifier
- ``create_model()``  – factory function
- ``list_backbones()`` / ``get_backbone_config()`` – registry helpers

Model definitions match the notebook (Cells 25, 29).
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


# ── LSTM ─────────────────────────────────────────────────────────────────────

class myLSTM(nn.Module):
    """Bidirectional LSTM text classifier.

    Architecture (from notebook Cell 25):
        Embedding → Bi-LSTM (stacked) → Dropout → FC → logits
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text : (batch, seq_len) – padded token indices
        lengths : (batch,) – actual sequence lengths
        """
        embedded = self.dropout(self.embedding(text))  # (B, L, E)

        # Pack to ignore padding
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False,
        )
        packed_output, (hidden, _cell) = self.lstm(packed)

        # Concatenate final forward and backward hidden states
        # hidden shape: (n_layers * n_directions, batch, hidden_dim)
        hidden_fwd = hidden[-2, :, :]  # last layer forward
        hidden_bwd = hidden[-1, :, :]  # last layer backward
        combined = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (B, hidden*2)

        return self.fc(self.dropout(combined))


# ── Transformer ──────────────────────────────────────────────────────────────

class myTransformer(nn.Module):
    """BERT-based text classifier.

    Architecture (from notebook Cell 29):
        BERT backbone → [CLS] pooling → Dropout → FC → logits
    """

    def __init__(
        self,
        checkpoint: str = "bert-base-uncased",
        num_classes: int = 10,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        from transformers import AutoConfig, AutoModel  # type: ignore[import-untyped]

        config = AutoConfig.from_pretrained(checkpoint)
        self.bert = AutoModel.from_pretrained(checkpoint, config=config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

    def trainable_parameters(self) -> list:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))


# ── factory & registry ───────────────────────────────────────────────────────

def create_model(
    backbone: str,
    num_classes: int = 10,
    vocab_size: int | None = None,
    device: torch.device | None = None,
    **kwargs,
) -> nn.Module:
    """Create a text classification model by backbone key.

    Parameters
    ----------
    backbone : ``"lstm"`` or ``"bert"``
    vocab_size : required for LSTM
    """
    if backbone == "lstm":
        if vocab_size is None:
            raise ValueError("vocab_size is required for LSTM backbone")
        model = myLSTM(
            vocab_size=vocab_size,
            embedding_dim=kwargs.get("embedding_dim", 128),
            hidden_dim=kwargs.get("hidden_dim", 256),
            output_dim=num_classes,
            n_layers=kwargs.get("n_layers", 2),
            bidirectional=kwargs.get("bidirectional", True),
            dropout=kwargs.get("dropout", 0.3),
            pad_idx=kwargs.get("pad_idx", 0),
        )
    elif backbone == "bert":
        model = myTransformer(
            checkpoint=kwargs.get("checkpoint", "bert-base-uncased"),
            num_classes=num_classes,
            dropout_rate=kwargs.get("dropout_rate", 0.3),
            freeze_backbone=kwargs.get("freeze_backbone", False),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone!r}. Available: lstm, bert")

    if device is not None:
        model = model.to(device)
    return model


def list_backbones() -> List[str]:
    return ["lstm", "bert"]


def get_backbone_config(backbone: str) -> Dict[str, str]:
    if backbone == "lstm":
        return {
            "key": "lstm",
            "display_name": "Bidirectional LSTM",
            "family": "rnn",
            "pretrained_source": "Random (embedding trained from scratch)",
            "description": "2-layer Bi-LSTM with 256 hidden units, 128-dim embeddings",
        }
    if backbone == "bert":
        return {
            "key": "bert",
            "display_name": "BERT Base Uncased",
            "family": "transformer",
            "pretrained_source": "HuggingFace / bert-base-uncased",
            "description": "12-layer Transformer, 768 hidden, ~110M params",
        }
    raise ValueError(f"Unsupported backbone: {backbone!r}")


BACKBONE_DISPLAY_NAMES = {
    "lstm": "Bidirectional LSTM",
    "bert": "BERT Base Uncased",
}


if __name__ == "__main__":
    # Quick smoke test
    for bb in list_backbones():
        cfg = get_backbone_config(bb)
        print(f"{bb:10s} → {cfg['display_name']} ({cfg['family']})")

    # Test LSTM creation
    lstm = create_model("lstm", num_classes=10, vocab_size=50002)
    dummy_text = torch.randint(0, 100, (4, 32))
    dummy_lengths = torch.tensor([32, 28, 15, 10])
    out = lstm(dummy_text, dummy_lengths)
    print(f"LSTM output: {tuple(out.shape)}")
