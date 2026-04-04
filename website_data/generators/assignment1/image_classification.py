from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT_DIR = Path(__file__).resolve().parents[3]
ASSIGNMENT1_DIR = ROOT_DIR / "assignment1"
IMAGE_CLASSIFICATION_DIR = ASSIGNMENT1_DIR / "image_classification"

if str(IMAGE_CLASSIFICATION_DIR) not in sys.path:
    sys.path.append(str(IMAGE_CLASSIFICATION_DIR))

from image_eda_reports import (
    FeatureExtractionConfig,
    _extract_resnet50_features_torchvision,
    build_gallery_data,
    build_similarity_outputs,
    collect_image_records,
    make_stratified_splits,
    sample_records_for_features,
)

FEATURE_CONFIG = FeatureExtractionConfig()
SIZE_SCATTER_SAMPLE = 3000
QUALITY_SCATTER_SAMPLE = 4000
RGB_MAX_IMAGES = 120
RGB_PIXELS_PER_IMAGE = 256
RGB_IMAGE_SIZE = 128
GALLERY_SAMPLES_PER_CLASS = 4
SIMILARITY_N_CLUSTERS = 5


def _core_sampling_notes(
    n_ok: int,
    n_scatter: int,
    n_quality_scatter: int,
    n_rgb_images: int,
    n_rgb_pixels: int,
) -> dict[str, list[str]]:
    return {
        "size_distribution": [
            f"Scatter (width × height): {n_scatter} images.",
            f"File size histogram: {n_ok} images.",
            f"Aspect ratio histogram: {n_ok} images.",
        ],
        "color_quality": [
            f"RGB 3D: {n_rgb_pixels} points from {n_rgb_images} images (resize {RGB_IMAGE_SIZE}×{RGB_IMAGE_SIZE}).",
            f"Sharpness vs contrast: {n_quality_scatter} images.",
            f"Sharpness histogram: {n_ok} images.",
            "Blur gallery: 10 images (5 blurriest, 5 sharpest) from the full set.",
            f"HSV: {n_quality_scatter} images (same rows as sharpness vs contrast).",
        ],
    }


def _classification_sampling_notes(
    total_images: int,
    train_count: int,
    val_count: int,
    test_count: int,
    n_feature_rows: int,
    feature_dim: int,
    n_clusters: int,
) -> dict[str, Any]:
    return {
        "sections": [
            {
                "title": "Overview",
                "lines": [f"Class counts: {total_images} images total."],
            },
            {
                "title": "Train / val / test",
                "lines": [
                    f"{train_count} train, {val_count} val, {test_count} test (stratified 70% / 15% / 15%).",
                ],
            },
            {
                "title": "Embeddings & 2D plots",
                "lines": [
                    f"ResNet50 features: {n_feature_rows} images.",
                    f"Input {FEATURE_CONFIG.image_size}×{FEATURE_CONFIG.image_size}, batch {FEATURE_CONFIG.batch_size}. PCA, t-SNE, and UMAP use this {n_feature_rows}×{feature_dim} matrix.",
                ],
            },
            {
                "title": "Similarity",
                "lines": [
                    f"Cosine similarity between class centroids from those features; {n_clusters} clusters; 15 pairs in the table.",
                ],
            },
        ],
    }


FOOD101_ROOT = IMAGE_CLASSIFICATION_DIR / "data" / "food-101"
IMG_DIR = FOOD101_ROOT / "images"
RECORDS_PARQUET = FOOD101_ROOT / "eda_records.parquet"
RESULTS_ROOT = IMAGE_CLASSIFICATION_DIR / "results" / "food101_classification"
EDA_OUTPUT = IMAGE_CLASSIFICATION_DIR / "eda" / "data"
CLASS_OUTPUT = IMAGE_CLASSIFICATION_DIR / "classification" / "data"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CORE_EDA_JSON = EDA_OUTPUT / "core_eda_data.json"
CLASSIFICATION_EDA_JSON = EDA_OUTPUT / "classification_eda_data.json"
RESULTS_JSON = CLASS_OUTPUT / "results_data.json"


np.random.seed(42)


def _to_abs_image_path(path_value: Any) -> Path:
    """Map parquet/records path values to absolute paths under IMG_DIR."""
    raw = Path(str(path_value))
    if raw.is_absolute():
        return raw.resolve()

    parts = raw.parts
    lower_parts = [part.lower() for part in parts]
    if "images" in lower_parts:
        idx = lower_parts.index("images")
        tail = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
        return (IMG_DIR / tail).resolve()

    return (IMG_DIR / raw).resolve()


def _save_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False)
    return path


def _try_reuse_built_json_outputs() -> list[Path] | None:
    """Reuse previously built website JSON outputs when they already exist and are valid."""
    outputs = [CORE_EDA_JSON, CLASSIFICATION_EDA_JSON, RESULTS_JSON]
    if not all(path.exists() for path in outputs):
        return None

    try:
        for path in outputs:
            with path.open("r", encoding="utf-8") as file_obj:
                json.load(file_obj)
    except Exception as exc:
        print(f"  Warning: existing JSON outputs are unreadable ({exc}); regenerating.")
        return None

    print("  Using existing built JSON outputs; skipping regeneration.")
    return outputs


def collect_all_image_paths() -> dict[str, list[Path]]:
    """Collect all image paths grouped by class."""
    class_images: dict[str, list[Path]] = {}
    for class_dir in sorted(IMG_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        class_images[class_dir.name] = images
    return class_images


def _normalize_records_df(df: pd.DataFrame) -> pd.DataFrame:
    """Align collect_image_records / parquet columns with website JSON (rel_path, dtypes)."""
    out = df.copy()
    out["path"] = out["path"].map(lambda p: str(_to_abs_image_path(p)))
    out["rel_path"] = out["path"].map(
        lambda p: str(Path(p).relative_to(ROOT_DIR)).replace("\\", "/")
    )
    out["file_size_kb"] = out["file_size_kb"].astype(float).round(1)
    out["aspect_ratio"] = out["aspect_ratio"].astype(float)
    out["width"] = out["width"].astype(int)
    out["height"] = out["height"].astype(int)
    return out


def _try_load_eda_records_parquet(class_images: dict[str, list[Path]]) -> pd.DataFrame | None:
    """Reuse data/food-101/eda_records.parquet from eda.ipynb when it matches the image tree."""
    if not RECORDS_PARQUET.exists():
        return None
    try:
        cached = pd.read_parquet(RECORDS_PARQUET)
    except Exception as exc:
        print(f"  Warning: could not read {RECORDS_PARQUET}: {exc}")
        return None
    required = {
        "path",
        "class_name",
        "width",
        "height",
        "file_size_kb",
        "aspect_ratio",
        "brightness",
        "contrast",
        "sharpness",
    }
    if not required.issubset(cached.columns):
        missing = required - set(cached.columns)
        print(f"  Warning: eda_records.parquet missing columns: {sorted(missing)}")
        return None
    expected_paths = {str(p.resolve()) for paths in class_images.values() for p in paths}
    cached_paths = {str(_to_abs_image_path(p)) for p in cached["path"].astype(str)}
    if cached_paths != expected_paths:
        print(
            "  Note: eda_records.parquet does not match images on disk; "
            "using collect_image_records instead."
        )
        return None
    print(f"  Using {RECORDS_PARQUET.name} ({len(cached)} rows).")
    return _normalize_records_df(cached)


def resolve_eda_records_df(class_images: dict[str, list[Path]]) -> pd.DataFrame:
    """Load eda_records.parquet when valid; otherwise collect_image_records (same as eda.ipynb)."""
    cached = _try_load_eda_records_parquet(class_images)
    if cached is not None:
        return cached
    n = sum(len(paths) for paths in class_images.values())
    print(f"  Running collect_image_records for {n} images...")
    return _normalize_records_df(collect_image_records(IMG_DIR))


def generate_core_eda_data(class_images: dict[str, list[Path]], records_df: pd.DataFrame) -> dict:
    """Generate core EDA data for the Assignment 1 image pages."""
    print("Generating Assignment 1 image core EDA data...")

    all_paths = [path for paths in class_images.values() for path in paths]
    total_images = len(all_paths)
    num_classes = len(class_images)

    n_ok = len(records_df)
    if n_ok == 0:
        raise RuntimeError("No images could be read for core EDA.")

    scat = records_df.sample(n=min(SIZE_SCATTER_SAMPLE, n_ok), random_state=42)
    qual = records_df.sample(n=min(QUALITY_SCATTER_SAMPLE, n_ok), random_state=42)
    rgb_pick = records_df.sample(n=min(RGB_MAX_IMAGES, n_ok), random_state=42)

    hist_file_sizes_kb = records_df["file_size_kb"].astype(float).round(1).tolist()
    hist_aspect_ratios = records_df["aspect_ratio"].tolist()
    sharpness_hist = [round(float(s), 4) for s in records_df["sharpness"].tolist()]

    blurriest = records_df.nsmallest(5, "sharpness")
    sharpest = records_df.nlargest(5, "sharpness")
    blur_extremes = {
        "blurriest": [
            {
                "sharpness": round(float(row["sharpness"]), 4),
                "class": row["class_name"],
                "path": row["rel_path"],
            }
            for _, row in blurriest.iterrows()
        ],
        "sharpest": [
            {
                "sharpness": round(float(row["sharpness"]), 4),
                "class": row["class_name"],
                "path": row["rel_path"],
            }
            for _, row in sharpest.iterrows()
        ],
    }

    hue_means: list[float] = []
    sat_means: list[float] = []
    val_means: list[float] = []
    for _, row in qual.iterrows():
        try:
            with Image.open(row["path"]) as img:
                hsv = img.convert("HSV")
                hsv_arr = np.asarray(hsv, dtype=np.float32)
                hue_means.append(round(float(hsv_arr[:, :, 0].mean()), 2))
                sat_means.append(round(float(hsv_arr[:, :, 1].mean()), 2))
                val_means.append(round(float(hsv_arr[:, :, 2].mean()), 2))
        except Exception:
            hue_means.append(0.0)
            sat_means.append(0.0)
            val_means.append(0.0)

    rgb_points: dict = {"r": [], "g": [], "b": [], "class_name": [], "brightness": []}
    pixel_rng = np.random.default_rng(FEATURE_CONFIG.random_state)
    for _, row in rgb_pick.iterrows():
        try:
            with Image.open(row["path"]) as img:
                rgb = img.convert("RGB").resize((RGB_IMAGE_SIZE, RGB_IMAGE_SIZE))
                pixels = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
            count = len(pixels)
            if count > RGB_PIXELS_PER_IMAGE:
                indices = pixel_rng.choice(count, size=RGB_PIXELS_PER_IMAGE, replace=False)
            else:
                indices = np.arange(count)
            for pixel_idx in indices:
                r, g, b = [int(pixels[pixel_idx, channel]) for channel in range(3)]
                rgb_points["r"].append(r)
                rgb_points["g"].append(g)
                rgb_points["b"].append(b)
                rgb_points["class_name"].append(row["class_name"])
                rgb_points["brightness"].append(round((r + g + b) / 3.0, 1))
        except Exception:
            pass

    gallery_df = records_df[["class_name", "path"]].copy()
    gallery_data = build_gallery_data(
        gallery_df,
        samples_per_class=GALLERY_SAMPLES_PER_CLASS,
        random_state=FEATURE_CONFIG.random_state,
    )
    gallery_samples: dict[str, list[str]] = {}
    for cls_name, items in gallery_data.items():
        gallery_samples[cls_name] = [
            str(Path(entry["path"]).relative_to(ROOT_DIR)).replace("\\", "/") for entry in items
        ]

    mean_w_pop = round(float(records_df["width"].mean()), 1)
    mean_h_pop = round(float(records_df["height"].mean()), 1)
    mean_ar_pop = round(float(records_df["aspect_ratio"].mean()), 3)

    return {
        "overview": {
            "total_images": total_images,
            "num_classes": num_classes,
            "total_size_mb": round(float(records_df["file_size_kb"].sum()) / 1024.0, 1),
            "mean_width": mean_w_pop,
            "mean_height": mean_h_pop,
            "min_width": int(records_df["width"].min()),
            "max_width": int(records_df["width"].max()),
            "min_height": int(records_df["height"].min()),
            "max_height": int(records_df["height"].max()),
            "mean_aspect_ratio": mean_ar_pop,
            "mean_sharpness": round(float(records_df["sharpness"].mean()), 4),
            "mean_brightness": round(float(records_df["brightness"].mean()), 4),
        },
        "image_sizes": {
            "widths": scat["width"].astype(int).tolist(),
            "heights": scat["height"].astype(int).tolist(),
            "file_sizes_kb": hist_file_sizes_kb,
            "aspect_ratios": hist_aspect_ratios,
            "class_names": scat["class_name"].tolist(),
        },
        "quality": {
            "brightness": [round(float(x), 4) for x in qual["brightness"].tolist()],
            "contrast": [round(float(x), 4) for x in qual["contrast"].tolist()],
            "sharpness": [round(float(x), 4) for x in qual["sharpness"].tolist()],
            "class_names": qual["class_name"].tolist(),
        },
        "hsv": {
            "hue": hue_means,
            "saturation": sat_means,
            "value": val_means,
            "class_names": qual["class_name"].tolist(),
        },
        "rgb_distribution": rgb_points,
        "blur_extremes": blur_extremes,
        "gallery_samples": gallery_samples,
        "sharpness_histogram": sharpness_hist,
        "sampling_notes": _core_sampling_notes(
            n_ok,
            len(scat),
            len(qual),
            len(rgb_pick),
            len(rgb_points["r"]),
        ),
    }


def generate_classification_eda_data(
    class_images: dict[str, list[Path]], records_df: pd.DataFrame
) -> dict:
    """Generate image classification EDA JSON for Assignment 1."""
    print("Generating Assignment 1 image classification EDA data...")

    class_counts = {name: len(paths) for name, paths in class_images.items()}
    sorted_classes = sorted(class_counts.items(), key=lambda item: -item[1])

    total = sum(class_counts.values())

    records = records_df[["class_name", "path"]].copy()
    split_df = make_stratified_splits(records, random_state=FEATURE_CONFIG.random_state)
    overall_counts = split_df["split"].value_counts().reindex(["train", "val", "test"]).fillna(0).astype(int)
    train_count = int(overall_counts["train"])
    val_count = int(overall_counts["val"])
    test_count = int(overall_counts["test"])

    class_splits: dict[str, dict[str, int]] = {}
    for cls_name in sorted(split_df["class_name"].unique()):
        sub = split_df[split_df["class_name"] == cls_name]["split"].value_counts()
        class_splits[cls_name] = {
            "train": int(sub.get("train", 0)),
            "val": int(sub.get("val", 0)),
            "test": int(sub.get("test", 0)),
        }
    feature_records = sample_records_for_features(
        records,
        sample_per_class=FEATURE_CONFIG.sample_per_class,
        max_images=FEATURE_CONFIG.max_images,
        random_state=FEATURE_CONFIG.random_state,
    )

    print("Extracting ResNet50 features for embeddings and similarity...")
    sampled_paths_str = feature_records["path"].tolist()
    sampled_labels = feature_records["class_name"].tolist()

    import torch

    original_device = torch.device
    if not torch.cuda.is_available():
        torch.device = lambda device_name: original_device("cpu") if device_name == "cuda" else original_device(device_name)

    try:
        features = _extract_resnet50_features_torchvision(
            sampled_paths_str,
            image_size=FEATURE_CONFIG.image_size,
            batch_size=FEATURE_CONFIG.batch_size,
        )
    finally:
        torch.device = original_device

    print("Computing PCA, t-SNE, and UMAP...")
    pca_2d = PCA(n_components=2, random_state=FEATURE_CONFIG.random_state)
    pca_pts = pca_2d.fit_transform(features)

    n_feat = len(features)
    tsne_perplexity = min(30, max(2, n_feat // 4), n_feat - 1)
    tsne = TSNE(
        n_components=2,
        random_state=FEATURE_CONFIG.random_state,
        init="pca",
        perplexity=tsne_perplexity,
    )
    tsne_pts = tsne.fit_transform(features)

    if FEATURE_CONFIG.use_umap:
        import umap

        reducer = umap.UMAP(random_state=FEATURE_CONFIG.random_state)
        umap_pts = reducer.fit_transform(features)
        umap_x = [round(float(value), 3) for value in umap_pts[:, 0]]
        umap_y = [round(float(value), 3) for value in umap_pts[:, 1]]
    else:
        umap_x = []
        umap_y = []

    pca_full = PCA().fit(features)
    variance_ratios = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratios)

    print("Computing class similarity outputs...")
    sim_df, top_pairs, cluster_df = build_similarity_outputs(
        features,
        feature_records,
        n_clusters=SIMILARITY_N_CLUSTERS,
    )

    n_clusters_used = int(cluster_df["cluster"].nunique())
    feature_dim = int(features.shape[1])
    sampling_notes = _classification_sampling_notes(
        total,
        train_count,
        val_count,
        test_count,
        len(feature_records),
        feature_dim,
        n_clusters_used,
    )

    return {
        "overview": {
            "total_images": total,
            "num_classes": len(class_counts),
            "max_count": max(class_counts.values()),
            "min_count": min(class_counts.values()),
            "imbalance_ratio": round(max(class_counts.values()) / max(min(class_counts.values()), 1), 2),
        },
        "class_distribution": {
            "classes": [class_name for class_name, _ in sorted_classes],
            "counts": [count for _, count in sorted_classes],
        },
        "split": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "train_pct": round(train_count / total * 100, 1),
            "val_pct": round(val_count / total * 100, 1),
            "test_pct": round(test_count / total * 100, 1),
        },
        "class_splits": class_splits,
        "pca": {
            "variance_ratios": [round(float(value), 4) for value in variance_ratios[:50]],
            "cumulative_variance": [round(float(value), 4) for value in cumulative_variance[:50]],
            "components": list(range(1, min(50, len(variance_ratios)) + 1)),
        },
        "embeddings": {
            "labels": sampled_labels,
            "pca_x": [round(float(value), 3) for value in pca_pts[:, 0]],
            "pca_y": [round(float(value), 3) for value in pca_pts[:, 1]],
            "tsne_x": [round(float(value), 3) for value in tsne_pts[:, 0]],
            "tsne_y": [round(float(value), 3) for value in tsne_pts[:, 1]],
            "umap_x": umap_x,
            "umap_y": umap_y,
        },
        "similarity": {
            "classes": sim_df.columns.tolist(),
            "matrix": sim_df.values.round(3).tolist(),
            "top_pairs": top_pairs.head(15).to_dict("records"),
            "clusters": cluster_df.groupby("cluster")["class_name"].apply(list).to_dict(),
        },
        "sampling_notes": sampling_notes,
    }


def generate_classification_results_data() -> dict:
    """Consolidate Assignment 1 image experiment results for the website."""
    print("Generating Assignment 1 model results data...")

    experiments: dict[str, dict] = {}
    learning_curves: dict[str, dict] = {}

    for exp_dir in sorted(RESULTS_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue

        summary_path = exp_dir / "experiment_summary.json"
        history_path = exp_dir / "history.csv"
        if not summary_path.exists():
            continue

        with summary_path.open("r", encoding="utf-8") as file_obj:
            summary = json.load(file_obj)
        experiments[summary["experiment_name"]] = summary

        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as file_obj:
                rows = list(csv.DictReader(file_obj))
            learning_curves[summary["experiment_name"]] = {
                "epochs": [int(row["epoch"]) for row in rows],
                "train_loss": [float(row["train_loss"]) for row in rows],
                "train_accuracy": [float(row["train_accuracy"]) for row in rows],
                "train_f1": [float(row["train_f1"]) for row in rows],
                "val_loss": [float(row["val_loss"]) for row in rows],
                "val_accuracy": [float(row["val_accuracy"]) for row in rows],
                "val_f1": [float(row["val_f1"]) for row in rows],
            }

    data = {"experiments": {}, "learning_curves": learning_curves}
    for name, exp in experiments.items():
        data["experiments"][name] = {
            "backbone": exp["backbone"],
            "augmentation": exp["augmentation"],
            "best_epoch": exp["best_epoch"],
            "training_time_minutes": round(exp["training_time_minutes"], 2),
            "test_metrics": exp["test_metrics"],
            "val_metrics": exp["val_metrics"],
            "resource_metrics": exp["resource_metrics"],
            "model_config": exp["model_config"],
            "data_config": exp["data_config"],
            "split_sizes": exp["split_sizes"],
            "class_names": exp.get("class_names", []),
        }

    return data


def generate_assignment1_image_classification_website_data() -> list[Path]:
    """Generate all website data artifacts used by Assignment 1 image pages."""
    print("=" * 60)
    print("Generating website data for Assignment 1 / Image Classification")
    print("=" * 60)

    reused_outputs = _try_reuse_built_json_outputs()
    if reused_outputs is not None:
        for output in reused_outputs:
            print(f"Reused: {output}")
        print("=" * 60)
        print("Done! Assignment 1 image website data reused successfully.")
        return reused_outputs

    class_images = collect_all_image_paths()
    print(f"Found {len(class_images)} classes, {sum(len(paths) for paths in class_images.values())} total images")

    records_df = resolve_eda_records_df(class_images)
    core_eda = generate_core_eda_data(class_images, records_df)
    classification_eda = generate_classification_eda_data(class_images, records_df)
    results_data = generate_classification_results_data()

    outputs = [
        _save_json(CORE_EDA_JSON, core_eda),
        _save_json(CLASSIFICATION_EDA_JSON, classification_eda),
        _save_json(RESULTS_JSON, results_data),
    ]

    for output in outputs:
        print(f"Saved: {output}")

    print("=" * 60)
    print("Done! Assignment 1 image website data generated successfully.")
    return outputs
