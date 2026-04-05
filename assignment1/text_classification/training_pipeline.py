"""Full training pipeline for Yahoo Answers text classification.

Mirrors ``assignment1/image_classification/training_pipeline.py``:
- ``ExperimentConfig`` dataclass with all hyperparameters
- ``fit_single_experiment()`` – train/eval/save loop
- ``compute_metrics()`` / ``run_lstm_epoch()`` / ``run_transformer_epoch()``
- ``profile_model()`` / ``benchmark_inference_time()``
- ``build_report_tables()`` / ``export_website_comparison_payload()``

Produces identical output structure to the image pipeline:
    results/yahoo_answers_classification/{backbone}_default/
        ├── best_model.pth
        ├── history.csv
        ├── confusion_matrix.csv
        ├── classification_report.csv
        └── experiment_summary.json
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import create_dataloaders
from models import BACKBONE_DISPLAY_NAMES, create_model, get_backbone_config

PLOT_METRIC_LABELS = {
    "loss": "Loss",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
}


@dataclass
class ExperimentConfig:
    """Training hyper-parameters for text classification."""

    # Common
    max_epochs: int = 10
    patience: int = 3
    min_delta: float = 1e-3
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    benchmark_runs: int = 50
    warmup_runs: int = 5
    results_root: str = "results/yahoo_answers_classification"
    num_workers: int = 2

    # LSTM defaults
    lstm_batch_size: int = 64
    lstm_lr: float = 1e-3
    lstm_max_seq_len: int = 256
    lstm_vocab_size: int = 50_000
    lstm_embedding_dim: int = 128
    lstm_hidden_dim: int = 256
    lstm_n_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_epochs: int = 3

    # Transformer defaults
    transformer_batch_size: int = 192
    transformer_lr: float = 1.6e-4
    transformer_max_len: int = 128
    transformer_checkpoint: str = "bert-base-uncased"
    transformer_dropout: float = 0.2
    transformer_freeze_backbone: bool = False
    transformer_epochs: int = 3

    @property
    def output_dir(self) -> Path:
        return Path(self.results_root)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_experiment_name(backbone: str) -> str:
    return BACKBONE_DISPLAY_NAMES.get(backbone, backbone)


def experiment_slug(backbone: str) -> str:
    return f"{backbone}_default"


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, loss_value: float) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0,
    )
    return {
        "loss": float(loss_value),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ── epoch runners ────────────────────────────────────────────────────────────

def run_lstm_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    desc: str = "",
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []

    for padded, lengths, labels in tqdm(loader, desc=desc, leave=False):
        padded = padded.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(padded, lengths)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(preds.cpu().tolist())

    metrics = compute_metrics(
        all_targets, all_predictions, total_loss / max(total_samples, 1),
    )
    return {
        "metrics": metrics,
        "targets": np.asarray(all_targets, dtype=np.int64),
        "predictions": np.asarray(all_predictions, dtype=np.int64),
    }


def run_transformer_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    desc: str = "",
) -> dict:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []

    for input_ids, attention_mask, labels in tqdm(loader, desc=desc, leave=False):
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits.detach(), dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(preds.cpu().tolist())

    metrics = compute_metrics(
        all_targets, all_predictions, total_loss / max(total_samples, 1),
    )
    return {
        "metrics": metrics,
        "targets": np.asarray(all_targets, dtype=np.int64),
        "predictions": np.asarray(all_predictions, dtype=np.int64),
    }


# ── profiling ────────────────────────────────────────────────────────────────

def profile_model(model: nn.Module) -> dict:
    return {
        "parameters": float(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": float(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }


def benchmark_inference_time(
    model: nn.Module,
    sample_batch: tuple,
    device: torch.device,
    config: ExperimentConfig,
    backbone: str,
) -> float:
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(config.warmup_runs):
            if backbone == "lstm":
                padded, lengths, _ = sample_batch
                model(padded[:1].to(device), lengths[:1].to(device))
            else:
                input_ids, attention_mask, _ = sample_batch
                model(input_ids=input_ids[:1].to(device), attention_mask=attention_mask[:1].to(device))

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(config.benchmark_runs):
            if backbone == "lstm":
                padded, lengths, _ = sample_batch
                model(padded[:1].to(device), lengths[:1].to(device))
            else:
                input_ids, attention_mask, _ = sample_batch
                model(input_ids=input_ids[:1].to(device), attention_mask=attention_mask[:1].to(device))

        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0 / config.benchmark_runs


# ── save utility ─────────────────────────────────────────────────────────────

def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ── main experiment runner ───────────────────────────────────────────────────

def fit_single_experiment(
    backbone: str,
    config: ExperimentConfig,
    device: torch.device,
    reuse_if_exists: bool = True,
) -> dict:
    """Train a single text classification experiment end-to-end.

    Produces the same output structure as the image classification pipeline.
    """
    set_seed(config.seed)
    experiment_name = format_experiment_name(backbone)
    slug = experiment_slug(backbone)
    output_dir = config.output_dir / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.csv"
    confusion_path = output_dir / "confusion_matrix.csv"
    report_path = output_dir / "classification_report.csv"
    summary_path = output_dir / "experiment_summary.json"

    # Reuse existing results if available
    if reuse_if_exists and all(
        p.exists() for p in (checkpoint_path, history_path, confusion_path, report_path, summary_path)
    ):
        with summary_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {
            "metadata": metadata,
            "history": pd.read_csv(history_path),
            "confusion_matrix": pd.read_csv(confusion_path, index_col=0),
            "classification_report": pd.read_csv(report_path, index_col=0),
        }

    # ── build dataloaders ────────────────────────────────────────────────
    tokenizer = None
    if backbone == "lstm":
        batch_size = config.lstm_batch_size
        lr = config.lstm_lr
        max_epochs = config.lstm_epochs
    else:
        batch_size = config.transformer_batch_size
        lr = config.transformer_lr
        max_epochs = config.transformer_epochs
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        tokenizer = AutoTokenizer.from_pretrained(config.transformer_checkpoint)

    max_seq_len = config.lstm_max_seq_len if backbone == "lstm" else config.transformer_max_len

    dataloaders = create_dataloaders(
        model_type="lstm" if backbone == "lstm" else "transformer",
        batch_size=batch_size,
        num_workers=config.num_workers,
        max_seq_len=max_seq_len,
        vocab_size=config.lstm_vocab_size,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
        tokenizer=tokenizer,
    )

    # ── build model ──────────────────────────────────────────────────────
    model_kwargs = {}
    if backbone == "lstm":
        model_kwargs.update(
            vocab_size=dataloaders["vocab_size"],
            embedding_dim=config.lstm_embedding_dim,
            hidden_dim=config.lstm_hidden_dim,
            n_layers=config.lstm_n_layers,
            dropout=config.lstm_dropout,
        )
    else:
        model_kwargs.update(
            checkpoint=config.transformer_checkpoint,
            dropout_rate=config.transformer_dropout,
            freeze_backbone=config.transformer_freeze_backbone,
        )

    model = create_model(
        backbone=backbone,
        num_classes=dataloaders["num_classes"],
        device=device,
        **model_kwargs,
    )

    resource_metrics = profile_model(model)

    # ── training loop ────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss().to(device)

    if backbone == "lstm":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        run_epoch = run_lstm_epoch
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        run_epoch = run_transformer_epoch

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_val_f1 = -1.0
    best_val_metrics = None
    stale_epochs = 0
    history_rows = []
    train_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_result = run_epoch(
            model, dataloaders["train_loader"], criterion, device,
            optimizer=optimizer, desc=f"{slug} train {epoch:02d}",
        )
        val_result = run_epoch(
            model, dataloaders["val_loader"], criterion, device,
            desc=f"{slug} val {epoch:02d}",
        )

        tm = train_result["metrics"]
        vm = val_result["metrics"]
        history_rows.append({
            "epoch": epoch,
            "train_loss": tm["loss"],
            "train_accuracy": tm["accuracy"],
            "train_precision": tm["precision"],
            "train_recall": tm["recall"],
            "train_f1": tm["f1"],
            "val_loss": vm["loss"],
            "val_accuracy": vm["accuracy"],
            "val_precision": vm["precision"],
            "val_recall": vm["recall"],
            "val_f1": vm["f1"],
        })

        print(
            f"[{experiment_name}] epoch {epoch:02d}/{max_epochs} | "
            f"train_acc={tm['accuracy']:.4f} | "
            f"val_acc={vm['accuracy']:.4f} | "
            f"val_f1={vm['f1']:.4f}"
        )

        if vm["f1"] > best_val_f1 + config.min_delta:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_f1 = vm["f1"]
            best_val_metrics = vm
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    training_time_minutes = (time.perf_counter() - train_start) / 60.0
    torch.save(best_state, checkpoint_path)

    # ── test evaluation ──────────────────────────────────────────────────
    test_result = run_epoch(
        model, dataloaders["test_loader"], criterion, device,
        desc=f"{slug} test",
    )

    # ── inference benchmark ──────────────────────────────────────────────
    sample_batch = next(iter(dataloaders["test_loader"]))
    inference_time_ms = benchmark_inference_time(model, sample_batch, device, config, backbone)

    # ── reports ──────────────────────────────────────────────────────────
    class_names = dataloaders["class_names"]
    cm = confusion_matrix(
        test_result["targets"], test_result["predictions"],
        labels=list(range(len(class_names))),
    )
    report = classification_report(
        test_result["targets"], test_result["predictions"],
        target_names=class_names, output_dict=True, zero_division=0,
    )

    history_df = pd.DataFrame(history_rows)
    confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    report_df = pd.DataFrame(report).transpose()

    history_df.to_csv(history_path, index=False)
    confusion_df.to_csv(confusion_path)
    report_df.to_csv(report_path)

    metadata = {
        "experiment_name": experiment_name,
        "experiment_slug": slug,
        "backbone": backbone,
        "best_epoch": best_epoch,
        "split_sizes": dataloaders["split_sizes"],
        "class_names": class_names,
        "model_config": get_backbone_config(backbone),
        "training_time_minutes": training_time_minutes,
        "val_metrics": best_val_metrics,
        "test_metrics": test_result["metrics"],
        "resource_metrics": {
            "parameters": resource_metrics["parameters"],
            "trainable_parameters": resource_metrics["trainable_parameters"],
            "model_size_mb": checkpoint_path.stat().st_size / (1024 ** 2),
            "inference_time_ms": inference_time_ms,
        },
        "checkpoint_path": str(checkpoint_path),
    }
    save_json(summary_path, metadata)

    return {
        "metadata": metadata,
        "history": history_df,
        "confusion_matrix": confusion_df,
        "classification_report": report_df,
    }


# ── report utilities (for website) ───────────────────────────────────────────

def build_report_tables(experiments: dict) -> dict:
    perf_rows, resource_rows, model_rows = [], [], []
    seen = set()

    for exp in experiments.values():
        meta = exp["metadata"]
        tm = meta["test_metrics"]
        rm = meta["resource_metrics"]

        perf_rows.append({
            "Model": meta["experiment_name"],
            "Precision ↑": tm["precision"] * 100,
            "Recall ↑": tm["recall"] * 100,
            "F1-Score ↑": tm["f1"] * 100,
            "Accuracy ↑": tm["accuracy"] * 100,
        })
        resource_rows.append({
            "Model": meta["experiment_name"],
            "Training Time ↓ (min)": meta["training_time_minutes"],
            "Inference Time ↓ (ms)": rm["inference_time_ms"],
            "Parameters ↓ (M)": rm["parameters"] / 1e6,
            "Size ↓ (MB)": rm["model_size_mb"],
        })
        if meta["backbone"] not in seen:
            seen.add(meta["backbone"])
            model_rows.append({
                "Model": meta["experiment_name"],
                "Backbone Key": meta["backbone"],
                "Family": meta["model_config"]["family"],
                "Pretrained Source": meta["model_config"]["pretrained_source"],
                "Parameters (M)": rm["parameters"] / 1e6,
                "Trainable Parameters (M)": rm["trainable_parameters"] / 1e6,
            })

    return {
        "performance": pd.DataFrame(perf_rows).sort_values("Accuracy ↑", ascending=False).reset_index(drop=True),
        "resources": pd.DataFrame(resource_rows).sort_values("Inference Time ↓ (ms)").reset_index(drop=True),
        "model_config": pd.DataFrame(model_rows),
    }


def export_website_comparison_payload(experiments: dict, output_path: Path) -> dict:
    models_payload = {}
    for exp in experiments.values():
        meta = exp["metadata"]
        m = meta["test_metrics"]
        r = meta["resource_metrics"]
        models_payload[meta["experiment_name"]] = {
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1_score": float(m["f1"]),
            "accuracy": float(m["accuracy"]),
            "training_time": float(meta["training_time_minutes"]),
            "confusion_matrix": exp["confusion_matrix"].astype(int).values.tolist(),
            "inference_time": float(r["inference_time_ms"]),
            "parameters": float(r["parameters"] / 1e6),
            "model_size": float(r["model_size_mb"]),
        }

    payload = {
        "models": models_payload,
        "metadata": {
            "data_sources": {
                "training": "history.csv + experiment_summary.json",
                "testing": "classification_report.csv + confusion_matrix.csv",
            },
            "merge_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    save_json(output_path, payload)
    return payload


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train text classification models on Yahoo Answers")
    parser.add_argument("--backbone", nargs="+", default=["lstm", "bert"], choices=["lstm", "bert"])
    parser.add_argument("--results-root", default="results/yahoo_answers_classification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = ExperimentConfig(results_root=args.results_root)
    experiments = {}

    for bb in args.backbone:
        print(f"\n{'=' * 60}")
        print(f"Training: {format_experiment_name(bb)}")
        print(f"{'=' * 60}")
        result = fit_single_experiment(bb, config, device)
        experiments[bb] = result

    if experiments:
        tables = build_report_tables(experiments)
        print("\n" + "=" * 60)
        print("Performance Summary:")
        print(tables["performance"].to_string(index=False))

        comparison_path = config.output_dir / "model-classification-comparison-data.json"
        export_website_comparison_payload(experiments, comparison_path)
        print(f"\nSaved website comparison data → {comparison_path}")
