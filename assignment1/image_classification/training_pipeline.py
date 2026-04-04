import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from thop import profile as thop_profile
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_download import download_and_extract_food101
from dataset import create_dataloaders as create_dataset_dataloaders
from models import create_model, get_backbone_config


BACKBONE_DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "mobilenet_v3": "MobileNetV3 Large 100",
    "vit_b16": "ViT-B/16",
    "swin_b": "Swin-B",
}

PLOT_METRIC_LABELS = {
    "loss": "Loss",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-Score",
    "top5_accuracy": "Top-5 Accuracy",
}


@dataclass
class ExperimentConfig:
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-3
    max_epochs: int = 40
    patience: int = 5
    min_delta: float = 1e-3
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    benchmark_runs: int = 100
    warmup_runs: int = 10
    results_root: str = "results/food101_classification"

    @property
    def output_dir(self) -> Path:
        return Path(self.results_root)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_backbone_name(backbone: str) -> str:
    return BACKBONE_DISPLAY_NAMES[backbone]


def format_experiment_name(backbone: str, use_augmentation: bool) -> str:
    suffix = "Aug" if use_augmentation else "NoAug"
    return f"{format_backbone_name(backbone)} ({suffix})"


def experiment_slug(backbone: str, use_augmentation: bool) -> str:
    suffix = "aug" if use_augmentation else "no_aug"
    return f"{backbone}_{suffix}"


def resolve_data_config(backbone: str) -> dict:
    timm_name = get_backbone_config(backbone)["timm_name"]
    model = timm.create_model(timm_name, pretrained=False)
    cfg = timm.data.resolve_data_config({}, model=model)
    del model

    return {
        "input_size": tuple(cfg["input_size"]),
        "mean": tuple(cfg["mean"]),
        "std": tuple(cfg["std"]),
    }


def create_dataloaders(
    backbone: str,
    config: ExperimentConfig,
    use_augmentation: bool,
    device: torch.device,
) -> dict:
    data_config = resolve_data_config(backbone)
    dataloaders = create_dataset_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_augmentation=use_augmentation,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        mean=data_config["mean"],
        std=data_config["std"],
        device=device,
    )
    dataloaders["data_config"] = data_config
    dataloaders["dataset_root"] = download_and_extract_food101()
    return dataloaders


def compute_metrics(targets, predictions, top5_predictions, loss_value: float) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    top5_hits = [int(target in row) for target, row in zip(targets, top5_predictions)]

    return {
        "loss": float(loss_value),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "top5_accuracy": float(np.mean(top5_hits)),
    }


def run_epoch(
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
    all_top5 = []

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=1)
        predictions = probabilities.argmax(dim=1)
        top5_predictions = probabilities.topk(k=min(5, probabilities.shape[1]), dim=1).indices

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_top5.extend(top5_predictions.cpu().tolist())

    metrics = compute_metrics(
        all_targets,
        all_predictions,
        all_top5,
        total_loss / total_samples,
    )

    return {
        "metrics": metrics,
        "targets": np.asarray(all_targets, dtype=np.int64),
        "predictions": np.asarray(all_predictions, dtype=np.int64),
    }


def profile_model(model: nn.Module, input_size: tuple[int, int, int], device: torch.device) -> dict:
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, *input_size, device=device)
    macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
    model.train(was_training)

    return {
        "flops": float(macs * 2),
        "parameters": float(sum(param.numel() for param in model.parameters())),
        "trainable_parameters": float(
            sum(param.numel() for param in model.parameters() if param.requires_grad)
        ),
    }


def benchmark_inference_time(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    config: ExperimentConfig,
) -> float:
    model.eval()
    sample = sample.to(device)

    with torch.no_grad():
        for _ in range(config.warmup_runs):
            model(sample)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(config.benchmark_runs):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0 / config.benchmark_runs


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def fit_single_experiment(
    backbone: str,
    use_augmentation: bool,
    config: ExperimentConfig,
    device: torch.device,
    reuse_if_exists: bool = True,
) -> dict:
    dataloaders = create_dataloaders(backbone, config, use_augmentation, device=device)
    experiment_name = format_experiment_name(backbone, use_augmentation)
    slug = experiment_slug(backbone, use_augmentation)
    output_dir = config.output_dir / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.csv"
    confusion_path = output_dir / "confusion_matrix.csv"
    report_path = output_dir / "classification_report.csv"
    summary_path = output_dir / "experiment_summary.json"

    if reuse_if_exists and all(
        path.exists()
        for path in (checkpoint_path, history_path, confusion_path, report_path, summary_path)
    ):
        with summary_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        return {
            "metadata": metadata,
            "history": pd.read_csv(history_path),
            "confusion_matrix": pd.read_csv(confusion_path, index_col=0),
            "classification_report": pd.read_csv(report_path, index_col=0),
        }

    model = create_model(
        backbone=backbone,
        num_classes=dataloaders["num_classes"],
        pretrained=True,
        device=device,
    )
    resource_metrics = profile_model(model, dataloaders["data_config"]["input_size"], device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=config.learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_val_f1 = -1.0
    best_val_metrics = None
    stale_epochs = 0
    history_rows = []
    train_start = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        train_result = run_epoch(
            model,
            dataloaders["train_loader"],
            criterion,
            device,
            optimizer=optimizer,
            desc=f"{slug} train {epoch:02d}",
        )
        val_result = run_epoch(
            model,
            dataloaders["val_loader"],
            criterion,
            device,
            desc=f"{slug} val {epoch:02d}",
        )

        train_metrics = train_result["metrics"]
        val_metrics = val_result["metrics"]
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_top5_accuracy": train_metrics["top5_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_top5_accuracy": val_metrics["top5_accuracy"],
            }
        )

        if epoch % 5 == 0 or epoch == config.max_epochs:
            print(
                f"[{experiment_name}] epoch {epoch:02d}/{config.max_epochs} | "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f}"
            )

        if val_metrics["f1"] > best_val_f1 + config.min_delta:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_f1 = val_metrics["f1"]
            best_val_metrics = val_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    training_time_minutes = (time.perf_counter() - train_start) / 60.0
    torch.save(best_state, checkpoint_path)

    test_result = run_epoch(
        model,
        dataloaders["test_loader"],
        criterion,
        device,
        desc=f"{slug} test",
    )

    sample_images, _ = next(iter(dataloaders["test_loader"]))
    inference_time_ms = benchmark_inference_time(model, sample_images[:1], device, config)

    class_names = dataloaders["class_names"]
    confusion = confusion_matrix(
        test_result["targets"],
        test_result["predictions"],
        labels=list(range(len(class_names))),
    )
    report = classification_report(
        test_result["targets"],
        test_result["predictions"],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    history_df = pd.DataFrame(history_rows)
    confusion_df = pd.DataFrame(confusion, index=class_names, columns=class_names)
    report_df = pd.DataFrame(report).transpose()

    history_df.to_csv(output_dir / "history.csv", index=False)
    confusion_df.to_csv(output_dir / "confusion_matrix.csv")
    report_df.to_csv(output_dir / "classification_report.csv")

    metadata = {
        "experiment_name": experiment_name,
        "experiment_slug": slug,
        "backbone": backbone,
        "augmentation": "aug" if use_augmentation else "no_aug",
        "best_epoch": best_epoch,
        "split_sizes": dataloaders["split_sizes"],
        "class_names": class_names,
        "data_config": {
            "input_size": list(dataloaders["data_config"]["input_size"]),
            "mean": list(dataloaders["data_config"]["mean"]),
            "std": list(dataloaders["data_config"]["std"]),
        },
        "model_config": get_backbone_config(backbone),
        "training_time_minutes": training_time_minutes,
        "val_metrics": best_val_metrics,
        "test_metrics": test_result["metrics"],
        "resource_metrics": {
            "flops": resource_metrics["flops"],
            "parameters": resource_metrics["parameters"],
            "trainable_parameters": resource_metrics["trainable_parameters"],
            "model_size_mb": checkpoint_path.stat().st_size / (1024**2),
            "inference_time_ms": inference_time_ms,
        },
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": dataloaders["dataset_root"],
    }
    save_json(output_dir / "experiment_summary.json", metadata)

    return {
        "metadata": metadata,
        "history": history_df,
        "confusion_matrix": confusion_df,
        "classification_report": report_df,
    }


def build_report_tables(experiments: dict) -> dict:
    performance_rows = []
    resource_rows = []
    model_rows = []
    seen_backbones = set()

    for experiment in experiments.values():
        meta = experiment["metadata"]
        test_metrics = meta["test_metrics"]
        resource_metrics = meta["resource_metrics"]

        performance_rows.append(
            {
                "Model": meta["experiment_name"],
                "Precision ↑": test_metrics["precision"] * 100,
                "Recall ↑": test_metrics["recall"] * 100,
                "F1-Score ↑": test_metrics["f1"] * 100,
                "Accuracy ↑": test_metrics["accuracy"] * 100,
                "Top-5 Acc ↑": test_metrics["top5_accuracy"] * 100,
            }
        )

        resource_rows.append(
            {
                "Model": meta["experiment_name"],
                "Training Time ↓ (min)": meta["training_time_minutes"],
                "Inference Time ↓ (ms)": resource_metrics["inference_time_ms"],
                "FLOPs ↓ (B)": resource_metrics["flops"] / 1e9,
                "Parameters ↓ (M)": resource_metrics["parameters"] / 1e6,
                "Model Size ↓ (MB)": resource_metrics["model_size_mb"],
            }
        )

        if meta["backbone"] not in seen_backbones:
            seen_backbones.add(meta["backbone"])
            model_rows.append(
                {
                    "Model": meta["experiment_name"].split(" (")[0],
                    "Backbone Key": meta["backbone"],
                    "Family": meta["model_config"]["family"],
                    "Pretrained Source": meta["model_config"]["pretrained_source"],
                    "timm Name": meta["model_config"]["timm_name"],
                    "Parameters (M)": resource_metrics["parameters"] / 1e6,
                    "Trainable Parameters (M)": resource_metrics["trainable_parameters"] / 1e6,
                }
            )

    performance_df = pd.DataFrame(performance_rows).sort_values("Accuracy ↑", ascending=False).reset_index(
        drop=True
    )
    resource_df = pd.DataFrame(resource_rows).sort_values("Inference Time ↓ (ms)").reset_index(drop=True)
    model_cfg_df = pd.DataFrame(model_rows)

    return {
        "performance": performance_df,
        "resources": resource_df,
        "model_config": model_cfg_df,
    }


def build_quick_summary(performance_df: pd.DataFrame, resource_df: pd.DataFrame) -> dict:
    best_row = performance_df.iloc[0]
    fastest_row = resource_df.iloc[0]

    return {
        "best_model": {
            "name": best_row["Model"],
            "accuracy": float(best_row["Accuracy ↑"]),
            "f1": float(best_row["F1-Score ↑"]),
        },
        "performance_range": {
            "accuracy_min": float(performance_df["Accuracy ↑"].min()),
            "accuracy_max": float(performance_df["Accuracy ↑"].max()),
            "f1_min": float(performance_df["F1-Score ↑"].min()),
            "f1_max": float(performance_df["F1-Score ↑"].max()),
            "top5_min": float(performance_df["Top-5 Acc ↑"].min()),
            "top5_max": float(performance_df["Top-5 Acc ↑"].max()),
        },
        "fastest_model": {
            "name": fastest_row["Model"],
            "inference_ms": float(fastest_row["Inference Time ↓ (ms)"]),
            "accuracy": float(
                performance_df.loc[performance_df["Model"] == fastest_row["Model"], "Accuracy ↑"].iloc[0]
            ),
        },
    }


def export_website_comparison_payload(experiments: dict, output_path: Path) -> dict:
    models_payload = {}
    for experiment in experiments.values():
        meta = experiment["metadata"]
        metrics = meta["test_metrics"]
        resources = meta["resource_metrics"]

        models_payload[meta["experiment_name"]] = {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1"]),
            "accuracy": float(metrics["accuracy"]),
            "top5_accuracy": float(metrics["top5_accuracy"]),
            "training_time": float(meta["training_time_minutes"]),
            "use_augmentation": bool(meta["augmentation"] == "aug"),
            "confusion_matrix": experiment["confusion_matrix"].astype(int).values.tolist(),
            "inference_time": float(resources["inference_time_ms"]),
            "evaluation_time": 0.0,
            "flops": float(resources["flops"]),
            "parameters": float(resources["parameters"] / 1e6),
            "model_size": float(resources["model_size_mb"]),
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


def create_learning_curve_figure(history_df: pd.DataFrame, experiment_name: str, metric: str = "accuracy"):
    train_col = f"train_{metric}"
    val_col = f"val_{metric}"
    plot_df = history_df[["epoch", train_col, val_col]].melt(
        id_vars="epoch",
        var_name="split",
        value_name="value",
    )
    plot_df["split"] = plot_df["split"].map({train_col: "Train", val_col: "Validation"})

    fig = px.line(
        plot_df,
        x="epoch",
        y="value",
        color="split",
        markers=True,
        title=f"{experiment_name} - {PLOT_METRIC_LABELS[metric]}",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Epoch",
        yaxis_title=PLOT_METRIC_LABELS[metric],
    )
    return fig


def create_confusion_matrix_figure(confusion_df: pd.DataFrame, experiment_name: str):
    fig = px.imshow(
        confusion_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title=f"{experiment_name} - Confusion Matrix",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
    )
    return fig
