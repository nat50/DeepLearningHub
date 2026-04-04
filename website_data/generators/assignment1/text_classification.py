from __future__ import annotations

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
ASSIGNMENT1_DIR = ROOT_DIR / "assignment1"
TEXT_CLASSIFICATION_DIR = ASSIGNMENT1_DIR / "text_classification"
EDA_OUTPUT = TEXT_CLASSIFICATION_DIR / "eda" / "data"
CLASS_OUTPUT = TEXT_CLASSIFICATION_DIR / "classification" / "data"


def _save_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    return path


def generate_text_eda_data() -> dict:
    """Template for Assignment 1 text EDA website data."""
    return {
        "status": "template",
        "assignment": "assignment1",
        "pipeline": "text_classification",
        "message": "TODO: replace this template with text EDA outputs.",
        "overview": {},
        "distributions": {},
        "embeddings": {},
        "samples": [],
    }


def generate_text_results_data() -> dict:
    """Template for Assignment 1 text model results website data."""
    return {
        "status": "template",
        "assignment": "assignment1",
        "pipeline": "text_classification",
        "message": "TODO: replace this template with text classification experiment summaries.",
        "experiments": {},
        "learning_curves": {},
        "evaluation": {},
    }


def generate_assignment1_text_classification_website_data() -> list[Path]:
    """Generate placeholder website data artifacts for Assignment 1 text classification."""
    print("=" * 60)
    print("Generating website data for Assignment 1 / Text Classification")
    print("=" * 60)
    print("Using template payloads. Update website_data/generators/assignment1/text_classification.py later.")

    outputs = [
        _save_json(EDA_OUTPUT / "text_eda_data.json", generate_text_eda_data()),
        _save_json(CLASS_OUTPUT / "results_data.json", generate_text_results_data()),
    ]

    for output in outputs:
        print(f"Saved template: {output}")

    print("=" * 60)
    print("Done! Assignment 1 text template data generated successfully.")
    return outputs
