from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
ASSIGNMENT1_DIR = ROOT_DIR / "assignment1"
MULTIMODAL_CLASSIFICATION_DIR = ASSIGNMENT1_DIR / "multimodal_classification"
EDA_OUTPUT = MULTIMODAL_CLASSIFICATION_DIR / "eda" / "data"/"chart"
CLASS_OUTPUT = MULTIMODAL_CLASSIFICATION_DIR / "classification" / "data"


def _save_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    return path


def generate_multimodal_eda_data() -> dict:
    """Generate multimodal EDA website data with chart metadata."""
    return {
        "status": "generated",
        "assignment": "assignment1",
        "pipeline": "multimodal_classification",
        "title": "Multimodal Classification EDA",
        "description": "Exploratory Data Analysis for product review images and text",
        "dataset": {
            "name": "Product Reviews (Image + Text)",
            "size": 7000,
            "categories": 21,
            "modalities": ["image", "text"]
        },
        "sections": {
            "image_properties": {
                "title": "Image Properties",
                "charts": [
                    {"file": "class_dist_bar.json", "title": "Category Distribution"},
                    {"file": "color_space.json", "title": "Color Space Analysis"},
                    {"file": "file_size.json", "title": "File Size Distribution"},
                    {"file": "quality_metrics.json", "title": "Quality Metrics"}
                ]
            },
            "text_analysis": {
                "title": "Text Analysis",
                "charts": [
                    {"file": "vocab_richness.json", "title": "Vocabulary Richness"},
                    {"file": "word_count_distribution.json", "title": "Word Count Distribution"},
                    {"file": "stop_words.json", "title": "Stop Words Analysis"}
                ]
            },
            "embeddings": {
                "title": "Dimensionality Reduction",
                "charts": [
                    {"file": "tsne_sampled_plot.json", "title": "t-SNE Visualization"},
                    {"file": "umap_sampled_plot.json", "title": "UMAP Visualization"},
                    {"file": "pca_variance.json", "title": "PCA Variance"}
                ]
            },
            "similarity": {
                "title": "Similarity Analysis",
                "charts": [
                    {"file": "similarity_matrix.json", "title": "Similarity Matrix"},
                    {"file": "similarity_heatmap.json", "title": "Similarity Heatmap"}
                ]
            }
        }
    }


def generate_multimodal_results_data() -> dict:
    """Generate multimodal classification results website data."""
    return {
        "status": "generated",
        "assignment": "assignment1",
        "pipeline": "multimodal_classification",
        "title": "Multimodal Classification Results",
        "description": "Zero-shot and few-shot classification results on product review data",
        "models": {
            "zero_shot": {
                "name": "Zero-Shot Classification",
                "description": "CLIP-based zero-shot learning",
                "metrics": {}
            },
            "few_shot": {
                "name": "Few-Shot Classification",
                "description": "Few-shot learning with metric learning",
                "metrics": {}
            }
        },
        "experiments": {},
        "learning_curves": {},
        "retrieval": {}
    }


def generate_assignment1_multimodal_classification_website_data() -> list[Path]:
    """Generate placeholder website data artifacts for Assignment 1 multimodal classification."""
    print("=" * 60)
    print("Generating website data for Assignment 1 / Multimodal Classification")
    print("=" * 60)
    print("Using template payloads. Update website_data/generators/assignment1/multimodal_classification.py later.")

    outputs = [
        _save_json(EDA_OUTPUT / "multimodal_eda_data.json", generate_multimodal_eda_data()),
        _save_json(CLASS_OUTPUT / "results_data.json", generate_multimodal_results_data()),
    ]

    for output in outputs:
        print(f"Saved template: {output}")

    print("=" * 60)
    print("Done! Assignment 1 multimodal template data generated successfully.")
    return outputs
