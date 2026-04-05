"""Generate website data artifacts for Assignment 1 text classification.

Reads experiment results from ``results/yahoo_answers_classification/``
(when they exist) and produces JSON files consumed by the HTML pages.

EDA data is hardcoded from notebook outputs since reprocessing 1.4M rows
would be slow and require the raw CSV on disk.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
ASSIGNMENT1_DIR = ROOT_DIR / "assignment1"
TEXT_CLASSIFICATION_DIR = ASSIGNMENT1_DIR / "text_classification"
RESULTS_ROOT = TEXT_CLASSIFICATION_DIR / "results" / "yahoo_answers_classification"
EDA_OUTPUT = TEXT_CLASSIFICATION_DIR / "eda" / "data"
CLASS_OUTPUT = TEXT_CLASSIFICATION_DIR / "classification" / "data"

CORE_EDA_JSON = EDA_OUTPUT / "text_eda_data.json"
RESULTS_JSON = CLASS_OUTPUT / "results_data.json"


def _save_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    return path


CLASS_NAMES = [
    "Society & Culture",
    "Science & Mathematics",
    "Health",
    "Education & Reference",
    "Computers & Internet",
    "Sports",
    "Business & Finance",
    "Entertainment & Music",
    "Family & Relationships",
    "Politics & Government",
]


def generate_text_eda_data() -> dict:
    """Generate EDA data from hardcoded notebook outputs.

    These values come directly from the executed notebook cells:
    - Cell 4: dataset overview (1.4M samples, 10 classes)
    - Cell 5: class distribution (140K each)
    - Cell 6: stopword statistics (50.25%)
    - Cell 7: word count distributions
    - Cell 9: post-cleaning statistics
    - Cell 10: vocabulary richness per category
    - Cell 22: per-class keywords
    """
    return {
        "overview": {
            "dataset_name": "Yahoo Answers Topic Classification",
            "total_samples": 1_400_000,
            "num_classes": 10,
            "num_columns": 4,
            "columns": ["class_index", "question_title", "question_content", "best_answer"],
            "missing_values": {
                "class_index": 0,
                "question_title": 0,
                "question_content": 631689,
                "best_answer": 24596,
            },
            "total_words_raw": 128_189_357,
            "avg_words_per_sample_raw": 91.56,
            "avg_words_per_sample_cleaned": 40.90,
            "total_words_after_cleaning": 57_258_160,
            "words_removed_pct": 55.3,
            "unique_meaningful_words": 2_162_990,
        },

        "class_distribution": {
            "classes": CLASS_NAMES,
            "counts": [140_000] * 10,
        },

        "stopword_stats": {
            "total_words": 128_189_357,
            "stop_words": 64_416_436,
            "stop_word_pct": 50.25,
            "unique_stop_words_found": 316,
        },

        "word_count_distribution": {
            "before_cleaning": {
                "mean": 91.56,
                "description": "Average words per sample before cleaning",
            },
            "after_cleaning": {
                "mean": 40.90,
                "description": "Average words per sample after stopword removal and cleaning",
            },
        },

        "vocabulary_richness": {
            "categories": [
                {"name": "Society & Culture",       "unique_words": 179621, "total_words": 6765800, "ttr": 0.0265, "samples": 140000},
                {"name": "Science & Mathematics",   "unique_words": 182383, "total_words": 6365025, "ttr": 0.0287, "samples": 140000},
                {"name": "Health",                  "unique_words": 147638, "total_words": 6825650, "ttr": 0.0216, "samples": 140000},
                {"name": "Education & Reference",   "unique_words": 206346, "total_words": 5612358, "ttr": 0.0368, "samples": 140000},
                {"name": "Computers & Internet",    "unique_words": 131064, "total_words": 5576682, "ttr": 0.0235, "samples": 140000},
                {"name": "Sports",                  "unique_words": 153819, "total_words": 4761439, "ttr": 0.0323, "samples": 140000},
                {"name": "Business & Finance",      "unique_words": 151676, "total_words": 5333581, "ttr": 0.0284, "samples": 140000},
                {"name": "Entertainment & Music",   "unique_words": 155283, "total_words": 4920750, "ttr": 0.0316, "samples": 140000},
                {"name": "Family & Relationships",  "unique_words": 119282, "total_words": 5888342, "ttr": 0.0203, "samples": 140000},
                {"name": "Politics & Government",   "unique_words": 175574, "total_words": 6192530, "ttr": 0.0284, "samples": 140000},
            ],
        },

        "top_keywords": {
            "Society & Culture":       ["people", "god", "like", "just", "dont"],
            "Science & Mathematics":   ["water", "like", "know", "does", "time"],
            "Health":                  ["like", "just", "dont", "know", "help"],
            "Education & Reference":   ["school", "know", "like", "help", "need"],
            "Computers & Internet":    ["computer", "use", "need", "want", "like"],
            "Sports":                  ["team", "game", "best", "year", "play"],
            "Business & Finance":      ["money", "pay", "work", "tax", "business"],
            "Entertainment & Music":   ["song", "like", "movie", "best", "know"],
            "Family & Relationships":  ["like", "just", "dont", "know", "love"],
            "Politics & Government":   ["people", "like", "think", "government", "war"],
        },

        "tokenization": {
            "lstm": {
                "vocab_size": 50002,
                "special_tokens": {"PAD": 0, "UNK": 1},
                "max_seq_len": 256,
            },
            "transformer": {
                "model": "bert-base-uncased",
                "vocab_size": 30522,
                "max_len": 256,
            },
        },
    }


def generate_text_results_data() -> dict:
    """Consolidate experiment results from results/ directory.

    Mirrors the image classification ``generate_classification_results_data()``.
    """
    experiments: dict[str, dict] = {}
    learning_curves: dict[str, dict] = {}

    if RESULTS_ROOT.exists():
        for exp_dir in sorted(RESULTS_ROOT.iterdir()):
            if not exp_dir.is_dir():
                continue

            summary_path = exp_dir / "experiment_summary.json"
            history_path = exp_dir / "history.csv"
            if not summary_path.exists():
                continue

            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)

            experiments[summary["experiment_name"]] = summary

            if history_path.exists():
                with history_path.open("r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                learning_curves[summary["experiment_name"]] = {
                    "epochs": [int(r["epoch"]) for r in rows],
                    "train_loss": [float(r["train_loss"]) for r in rows],
                    "train_accuracy": [float(r["train_accuracy"]) for r in rows],
                    "train_f1": [float(r["train_f1"]) for r in rows],
                    "val_loss": [float(r["val_loss"]) for r in rows],
                    "val_accuracy": [float(r["val_accuracy"]) for r in rows],
                    "val_f1": [float(r["val_f1"]) for r in rows],
                }

    data: dict[str, Any] = {
        "experiments": {},
        "learning_curves": learning_curves,
        "status": "ready" if experiments else "pending",
    }

    for name, exp in experiments.items():
        data["experiments"][name] = {
            "backbone": exp["backbone"],
            "best_epoch": exp["best_epoch"],
            "training_time_minutes": round(exp["training_time_minutes"], 2),
            "test_metrics": exp["test_metrics"],
            "val_metrics": exp["val_metrics"],
            "resource_metrics": exp["resource_metrics"],
            "model_config": exp["model_config"],
            "split_sizes": exp["split_sizes"],
            "class_names": exp.get("class_names", []),
        }

    # If no results exist, provide model info for the config tab
    if not experiments:
        data["model_info"] = {
            "lstm": {
                "display_name": "Bidirectional LSTM",
                "family": "rnn",
                "pretrained_source": "Random (embedding trained from scratch)",
                "architecture": "Embedding(50K, 128) → Bi-LSTM(256, 2 layers) → FC(512→10)",
                "hyperparameters": {
                    "batch_size": 64,
                    "learning_rate": "1e-3",
                    "max_epochs": 3,
                    "max_seq_len": 256,
                    "optimizer": "Adam",
                },
            },
            "bert": {
                "display_name": "BERT Base Uncased",
                "family": "transformer",
                "pretrained_source": "HuggingFace / bert-base-uncased",
                "architecture": "BERT(12 layers, 768 hidden) → Dropout(0.3) → FC(768→10)",
                "hyperparameters": {
                    "batch_size": 16,
                    "learning_rate": "2e-5",
                    "max_epochs": 2,
                    "max_len": 256,
                    "optimizer": "AdamW",
                },
            },
        }

    return data


def _try_reuse_built_json_outputs() -> list[Path] | None:
    """Reuse previously built JSON outputs if they exist and are valid."""
    outputs = [CORE_EDA_JSON, RESULTS_JSON]
    if not all(p.exists() for p in outputs):
        return None
    try:
        for p in outputs:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Check it's not the old template version
            if data.get("status") == "template":
                return None
    except Exception as exc:
        print(f"  Warning: existing JSON outputs are unreadable ({exc}); regenerating.")
        return None

    print("  Using existing built JSON outputs; skipping regeneration.")
    return outputs


def generate_assignment1_text_classification_website_data() -> list[Path]:
    """Generate all website data artifacts for Assignment 1 text classification."""
    print("=" * 60)
    print("Generating website data for Assignment 1 / Text Classification")
    print("=" * 60)

    reused = _try_reuse_built_json_outputs()
    if reused is not None:
        for p in reused:
            print(f"Reused: {p}")
        print("=" * 60)
        print("Done! Assignment 1 text website data reused successfully.")
        return reused

    eda_data = generate_text_eda_data()
    results_data = generate_text_results_data()

    outputs = [
        _save_json(CORE_EDA_JSON, eda_data),
        _save_json(RESULTS_JSON, results_data),
    ]

    for p in outputs:
        print(f"Saved: {p}")

    if results_data.get("status") == "pending":
        print("  Note: No training results found. Run training_pipeline.py first.")

    print("=" * 60)
    print("Done! Assignment 1 text website data generated successfully.")
    return outputs
