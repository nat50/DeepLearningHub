"""
Yahoo Answers Classification with BERT Fine-tuning
=============================================

This script fine-tunes various BERT models (BERT-base, DistilBERT, TinyBERT)
with different pooling strategies (CLS, Mean, Pooler) for text classification.

Models trained: 8 configurations
- BERT-base: pooler_output, cls_token, mean_pooling
- DistilBERT: cls_token, mean_pooling (no pooler)
- TinyBERT: pooler_output, cls_token, mean_pooling

Dataset: Yahoo Answers (10 categories, 1.4M samples)
- Train: 980,000 samples (70%)
- Validation: 210,000 samples (15%)
- Test: 210,000 samples (15%)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import json
import csv
import copy
import time
import random
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dataset import (
    create_dataloaders,
    CLASS_NAMES,
    TransformerDataset,
    make_transformer_collate_fn,
)


# ========== Configuration ==========

BERT_MODELS = {
    "bert_base": {
        "name": "BERT-base",
        "model_name": "bert-base-uncased",
        "params_millions": 110,
        "has_pooler": True,
    },
    "distilbert": {
        "name": "DistilBERT",
        "model_name": "distilbert-base-uncased",
        "params_millions": 66,
        "has_pooler": False,
    },
    "tinybert": {
        "name": "TinyBERT",
        "model_name": "prajjwal1/bert-tiny",
        "params_millions": 14,
        "has_pooler": True,
    },
}

POOLING_STRATEGIES = {
    "cls_token": "[CLS] Token",
    "mean_pooling": "Mean Pooling",
    "pooler_output": "Pooler Output",
}

TRAINING_CONFIG = {
    "max_length": 128,
    "batch_size": 16,
    "eval_batch_size": 32,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "seed": 42,
    "patience": 3,
    "min_delta": 1e-3,
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15,
    "num_workers": 0,
    "benchmark_runs": 50,
    "warmup_runs": 5,
}


# ========== Model Definition ==========


class BERTWithCustomPooling(nn.Module):
    """BERT model with custom pooling strategies for Yahoo Answers classification."""

    def __init__(
        self, model_name, pooling_strategy="cls_token", num_labels=10, dropout=0.1
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy

        # Load pre-trained BERT
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Check if model has pooler
        self.has_pooler = hasattr(self.bert, "pooler") and self.bert.pooler is not None

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if self.pooling_strategy == "cls_token":
            pooled = outputs.last_hidden_state[:, 0, :]

        elif self.pooling_strategy == "mean_pooling":
            last_hidden = outputs.last_hidden_state
            attention_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            sum_hidden = torch.sum(last_hidden * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask

        elif self.pooling_strategy == "pooler_output":
            if not self.has_pooler:
                raise ValueError("Model does not have pooler output")
            pooled = outputs.pooler_output

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ========== Utilities ==========


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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
) -> float:
    """Benchmark single-sample inference time in milliseconds."""
    model.eval()
    with torch.no_grad():
        input_ids, attention_mask, _ = sample_batch
        single_ids = input_ids[:1].to(device)
        single_mask = attention_mask[:1].to(device)

        # Warmup
        for _ in range(TRAINING_CONFIG["warmup_runs"]):
            model(single_ids, single_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(TRAINING_CONFIG["benchmark_runs"]):
            model(single_ids, single_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0 / TRAINING_CONFIG["benchmark_runs"]


# ========== Training ==========


class BERTFineTuner:
    """Train and evaluate BERT models with multiple pooling strategies."""

    def __init__(self, output_dir="results/yahoo_answers_classification/bert_finetuning"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n🖥️  Device: {self.device}")

        set_seed(TRAINING_CONFIG["seed"])

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer=None,
        desc: str = "",
    ) -> dict:
        """Run one training or evaluation epoch."""
        is_training = optimizer is not None
        model.train(is_training)

        total_loss = 0.0
        total_samples = 0
        all_targets = []
        all_predictions = []

        for input_ids, attention_mask, labels in tqdm(loader, desc=desc, leave=False):
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_training):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                if is_training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            preds = torch.argmax(logits.detach(), dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(preds.cpu().tolist())

        avg_loss = total_loss / max(total_samples, 1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted", zero_division=0
        )
        metrics = {
            "loss": float(avg_loss),
            "accuracy": float(accuracy_score(all_targets, all_predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        return {
            "metrics": metrics,
            "targets": np.asarray(all_targets, dtype=np.int64),
            "predictions": np.asarray(all_predictions, dtype=np.int64),
        }

    def train_combination(
        self,
        model_key: str,
        pooling_key: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        class_names: list,
        num_classes: int = 10,
        reuse_if_exists: bool = True,
    ) -> dict | None:
        """Train a single model + pooling combination."""

        model_info = BERT_MODELS[model_key]

        # Skip if pooling requires pooler but model doesn't have it
        if pooling_key == "pooler_output" and not model_info["has_pooler"]:
            print(f"\n⏭️  Skipping {model_key} + {pooling_key}: No pooler")
            return None

        combination_name = f"{model_key}_{pooling_key}"
        display_name = f"{model_info['name']} + {POOLING_STRATEGIES[pooling_key]}"

        exp_dir = self.output_dir / combination_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        summary_path = exp_dir / "experiment_summary.json"
        history_path = exp_dir / "history.csv"
        checkpoint_path = exp_dir / "best_model.pth"
        confusion_path = exp_dir / "confusion_matrix.csv"
        report_path = exp_dir / "classification_report.csv"

        # Reuse existing results if available
        if reuse_if_exists and all(
            p.exists()
            for p in (summary_path, history_path, checkpoint_path)
        ):
            print(f"\n✅ Reusing existing results for {display_name}")
            with summary_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        print(f"\n{'='*70}")
        print(f"🚀 Training: {display_name}")
        print(f"{'='*70}")

        # Initialize model
        model = BERTWithCustomPooling(
            model_info["model_name"],
            pooling_strategy=pooling_key,
            num_labels=num_classes,
        ).to(self.device)

        resource_metrics = profile_model(model)

        # Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )

        criterion = nn.CrossEntropyLoss().to(self.device)

        # Training loop with early stopping
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = 1
        best_val_f1 = -1.0
        best_val_metrics = None
        stale_epochs = 0
        history_rows = []
        train_start = time.perf_counter()

        for epoch in range(1, TRAINING_CONFIG["num_epochs"] + 1):
            train_result = self._run_epoch(
                model, train_loader, criterion,
                optimizer=optimizer,
                desc=f"{combination_name} train {epoch:02d}",
            )
            val_result = self._run_epoch(
                model, val_loader, criterion,
                desc=f"{combination_name} val {epoch:02d}",
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
                f"[{display_name}] epoch {epoch:02d}/{TRAINING_CONFIG['num_epochs']} | "
                f"train_acc={tm['accuracy']:.4f} | "
                f"val_acc={vm['accuracy']:.4f} | "
                f"val_f1={vm['f1']:.4f}"
            )

            if vm["f1"] > best_val_f1 + TRAINING_CONFIG["min_delta"]:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_val_f1 = vm["f1"]
                best_val_metrics = vm
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= TRAINING_CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_state)
        training_time_seconds = time.perf_counter() - train_start
        training_time_minutes = training_time_seconds / 60.0
        torch.save(best_state, checkpoint_path)

        # Save history
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

        # Test evaluation
        test_result = self._run_epoch(
            model, test_loader, criterion,
            desc=f"{combination_name} test",
        )

        # Inference benchmark
        sample_batch = next(iter(test_loader))
        inference_time_ms = benchmark_inference_time(model, sample_batch, self.device)

        # Confusion matrix and classification report
        cm = confusion_matrix(
            test_result["targets"],
            test_result["predictions"],
            labels=list(range(num_classes)),
        )
        report = classification_report(
            test_result["targets"],
            test_result["predictions"],
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        # Save confusion matrix
        import pandas as pd
        confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        confusion_df.to_csv(confusion_path)

        # Save classification report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(report_path)

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                test_result["targets"],
                test_result["predictions"],
                average=None,
                zero_division=0,
            )
        )
        per_class_metrics = [
            {
                "class": class_names[i],
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
            for i in range(num_classes)
        ]

        # Build summary
        model_size_mb = checkpoint_path.stat().st_size / (1024**2)

        summary = {
            "experiment_name": display_name,
            "experiment_slug": combination_name,
            "model_base": model_key,
            "model_base_name": model_info["name"],
            "model_checkpoint": model_info["model_name"],
            "pooling_strategy": pooling_key,
            "pooling_display": POOLING_STRATEGIES[pooling_key],
            "display_name": display_name,
            "best_epoch": best_epoch,
            "training_time_seconds": training_time_seconds,
            "training_time_minutes": training_time_minutes,
            "test_metrics": test_result["metrics"],
            "val_metrics": best_val_metrics,
            "resource_metrics": {
                "parameters": resource_metrics["parameters"],
                "trainable_parameters": resource_metrics["trainable_parameters"],
                "model_size_mb": model_size_mb,
                "inference_time_ms": inference_time_ms,
            },
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class_metrics,
            "class_names": class_names,
        }

        save_json(summary_path, summary)

        print(f"\n✅ {display_name} completed!")
        print(f"   Best Val F1: {best_val_f1:.4f}")
        print(f"   Test Acc: {test_result['metrics']['accuracy']:.4f}")
        print(f"   Train Time: {training_time_minutes:.1f} min")
        print(f"   Results saved to: {exp_dir}")

        return summary

    def train_all(self, reuse_if_exists: bool = True) -> list[dict]:
        """Train all 8 BERT configurations."""

        print("=" * 70)
        print("YAHOO ANSWERS CLASSIFICATION - BERT FINE-TUNING")
        print("=" * 70)

        # We need to create dataloaders for each model since tokenizers differ
        all_results = []

        for model_key, model_info in BERT_MODELS.items():
            # Load tokenizer for this model
            tokenizer = AutoTokenizer.from_pretrained(model_info["model_name"])

            print(f"\n📦 Loading data with {model_info['name']} tokenizer...")

            dataloaders = create_dataloaders(
                model_type="transformer",
                batch_size=TRAINING_CONFIG["batch_size"],
                num_workers=TRAINING_CONFIG["num_workers"],
                max_seq_len=TRAINING_CONFIG["max_length"],
                train_size=TRAINING_CONFIG["train_size"],
                val_size=TRAINING_CONFIG["val_size"],
                test_size=TRAINING_CONFIG["test_size"],
                seed=TRAINING_CONFIG["seed"],
                tokenizer=tokenizer,
            )

            for pooling_key in POOLING_STRATEGIES.keys():
                result = self.train_combination(
                    model_key,
                    pooling_key,
                    train_loader=dataloaders["train_loader"],
                    val_loader=dataloaders["val_loader"],
                    test_loader=dataloaders["test_loader"],
                    class_names=dataloaders["class_names"],
                    num_classes=dataloaders["num_classes"],
                    reuse_if_exists=reuse_if_exists,
                )
                if result is not None:
                    all_results.append(result)

        # Save combined results
        self._save_combined_results(all_results)

        # Summary
        print(f"\n{'='*70}")
        print("🎉 ALL TRAINING COMPLETED!")
        print(f"{'='*70}")
        print(f"\nTrained {len(all_results)} configurations:\n")

        for r in sorted(all_results, key=lambda x: x["test_metrics"]["accuracy"], reverse=True):
            acc = r["test_metrics"]["accuracy"] * 100
            print(f"  {r['display_name']:<35} Acc: {acc:.2f}%")

        return all_results

    def _save_combined_results(self, results: list[dict]):
        """Save combined results JSON for the website."""
        if not results:
            return

        # Build website payload
        experiments = {}
        learning_curves = {}

        for r in results:
            slug = r["experiment_slug"]
            experiments[r["display_name"]] = {
                "model_base": r["model_base"],
                "model_base_name": r["model_base_name"],
                "model_checkpoint": r["model_checkpoint"],
                "pooling_strategy": r["pooling_strategy"],
                "pooling_display": r["pooling_display"],
                "best_epoch": r["best_epoch"],
                "training_time_minutes": round(r["training_time_minutes"], 2),
                "training_time_seconds": round(r["training_time_seconds"], 1),
                "test_metrics": r["test_metrics"],
                "val_metrics": r["val_metrics"],
                "resource_metrics": r["resource_metrics"],
                "confusion_matrix": r["confusion_matrix"],
                "per_class_metrics": r["per_class_metrics"],
                "class_names": r["class_names"],
            }

            # Load learning curves from history.csv
            history_path = self.output_dir / slug / "history.csv"
            if history_path.exists():
                with history_path.open("r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                learning_curves[r["display_name"]] = {
                    "epochs": [int(row["epoch"]) for row in rows],
                    "train_loss": [float(row["train_loss"]) for row in rows],
                    "train_accuracy": [float(row["train_accuracy"]) for row in rows],
                    "train_f1": [float(row["train_f1"]) for row in rows],
                    "val_loss": [float(row["val_loss"]) for row in rows],
                    "val_accuracy": [float(row["val_accuracy"]) for row in rows],
                    "val_f1": [float(row["val_f1"]) for row in rows],
                }

        payload = {
            "experiments": experiments,
            "learning_curves": learning_curves,
            "status": "ready",
            "metadata": {
                "dataset": "Yahoo Answers",
                "num_classes": 10,
                "class_names": list(CLASS_NAMES.values()),
                "training_config": TRAINING_CONFIG,
                "models": {k: v for k, v in BERT_MODELS.items()},
                "pooling_strategies": POOLING_STRATEGIES,
                "timestamp": datetime.now().isoformat(),
            },
        }

        # Save to results dir
        save_json(self.output_dir / "bert_finetuning_results.json", payload)

        # Also save to website data dir
        website_data_dir = Path(__file__).parent / "bert_finetuning" / "data"
        save_json(website_data_dir / "bert_finetuning_results.json", payload)

        print(f"\n📊 Combined results saved to:")
        print(f"   {self.output_dir / 'bert_finetuning_results.json'}")
        print(f"   {website_data_dir / 'bert_finetuning_results.json'}")


# ========== Main ==========


def main():
    """Train all 8 BERT configurations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune BERT models on Yahoo Answers"
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=list(BERT_MODELS.keys()),
        choices=list(BERT_MODELS.keys()),
        help="Which models to train",
    )
    parser.add_argument(
        "--pooling",
        nargs="+",
        default=list(POOLING_STRATEGIES.keys()),
        choices=list(POOLING_STRATEGIES.keys()),
        help="Which pooling strategies to use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if results exist",
    )
    args = parser.parse_args()

    trainer = BERTFineTuner()

    # Filter to requested models/pooling
    if args.model != list(BERT_MODELS.keys()) or args.pooling != list(
        POOLING_STRATEGIES.keys()
    ):
        # Train specific combinations
        all_results = []
        for model_key in args.model:
            model_info = BERT_MODELS[model_key]
            tokenizer = AutoTokenizer.from_pretrained(model_info["model_name"])

            dataloaders = create_dataloaders(
                model_type="transformer",
                batch_size=TRAINING_CONFIG["batch_size"],
                num_workers=TRAINING_CONFIG["num_workers"],
                max_seq_len=TRAINING_CONFIG["max_length"],
                train_size=TRAINING_CONFIG["train_size"],
                val_size=TRAINING_CONFIG["val_size"],
                test_size=TRAINING_CONFIG["test_size"],
                seed=TRAINING_CONFIG["seed"],
                tokenizer=tokenizer,
            )

            for pooling_key in args.pooling:
                result = trainer.train_combination(
                    model_key,
                    pooling_key,
                    train_loader=dataloaders["train_loader"],
                    val_loader=dataloaders["val_loader"],
                    test_loader=dataloaders["test_loader"],
                    class_names=dataloaders["class_names"],
                    num_classes=dataloaders["num_classes"],
                    reuse_if_exists=not args.force,
                )
                if result:
                    all_results.append(result)

        trainer._save_combined_results(all_results)
    else:
        trainer.train_all(reuse_if_exists=not args.force)


if __name__ == "__main__":
    main()
