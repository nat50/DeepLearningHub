from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FeatureExtractionConfig:
    image_size: int = 224
    batch_size: int = 16
    max_images: int = 1200
    sample_per_class: int = 120
    random_state: int = 42
    use_umap: bool = True


def set_plot_style() -> None:
    px.defaults.template = "plotly_white"
    px.defaults.width = 900
    px.defaults.height = 500


def _read_image_rgb_uint8(path: str, image_size: Optional[int] = None) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if image_size is not None:
            rgb = rgb.resize((image_size, image_size))
        arr = np.asarray(rgb, dtype=np.uint8)
    return arr


def iter_image_paths(img_dir: Path) -> Iterable[Tuple[str, Path]]:
    for class_dir in sorted([path for path in img_dir.iterdir() if path.is_dir()]):
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield class_dir.name, path


def collect_image_records(
    img_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for class_name, path in iter_image_paths(img_dir):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            arr = np.asarray(rgb, dtype=np.float32) / 255.0

        gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        brightness = float(arr.mean())
        contrast = float(arr.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_rgb = arr.reshape(-1, 3).mean(axis=0)

        rows.append(
            {
                "folder_name": class_name,
                "class_name": class_name,
                "path": str(path),
                "file_size_kb": path.stat().st_size / 1024.0,
                "width": width,
                "height": height,
                "aspect_ratio": width / max(height, 1),
                "pixels": width * height,
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "mean_r": float(mean_rgb[0]),
                "mean_g": float(mean_rgb[1]),
                "mean_b": float(mean_rgb[2]),
            }
        )

    records = pd.DataFrame(rows)
    records["image_id"] = np.arange(len(records))
    return records


def summarize_core_dataset(records: pd.DataFrame) -> Dict[str, object]:
    summary = {
        "total_images": int(len(records)),
        "num_classes": int(records["class_name"].nunique()),
        "total_size_mb": float(records["file_size_kb"].sum() / 1024.0),
        "class_counts": (
            records.groupby("class_name")
            .size()
            .reset_index(name="image_count")
            .sort_values("image_count", ascending=False)
            .reset_index(drop=True)
        ),
    }
    return summary


def display_core_overview(summary: Dict[str, object]) -> None:
    print("CORE EDA REPORT")
    print(f"Total images: {summary['total_images']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Estimated dataset size: {summary['total_size_mb']:.2f} MB")
    display(summary["class_counts"])


def build_gallery_data(
    records: pd.DataFrame,
    classes: Optional[Sequence[str]] = None,
    samples_per_class: int = 4,
    random_state: int = 42,
) -> Dict[str, List[Dict[str, str]]]:
    rng = np.random.default_rng(random_state)
    classes = list(classes) if classes is not None else sorted(records["class_name"].unique())

    gallery_data: Dict[str, List[Dict[str, str]]] = {}
    for class_name in classes:
        class_rows = records.loc[records["class_name"] == class_name]
        chosen = class_rows.sample(
            n=min(samples_per_class, len(class_rows)),
            random_state=int(rng.integers(1_000_000)),
        )
        gallery_data[class_name] = [
            {
                "path": row["path"],
                "class_name": row["class_name"],
            }
            for _, row in chosen.iterrows()
        ]

    return gallery_data


def plot_core_size_analysis(records: pd.DataFrame) -> None:
    fig1 = px.scatter(
        records.sample(min(3000, len(records)), random_state=42),
        x="width",
        y="height",
        opacity=0.6,
        title="Image Size Distribution (Width vs Height)",
    )
    fig1.show()

    fig2 = px.histogram(
        records,
        x="file_size_kb",
        nbins=40,
        title="File Size Distribution (KB)",
        labels={"file_size_kb": "File size (KB)"},
    )
    fig2.show()

    fig3 = px.histogram(
        records,
        x="aspect_ratio",
        nbins=40,
        title="Aspect Ratio Distribution",
        labels={"aspect_ratio": "Width / Height"},
    )
    fig3.show()


def sample_rgb_distribution(
    records: pd.DataFrame,
    max_images: int = 120,
    pixels_per_image: int = 256,
    image_size: int = 128,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    sampled_records = records.sample(min(max_images, len(records)), random_state=random_state)

    sampled_frames: List[pd.DataFrame] = []
    for _, row in sampled_records.iterrows():
        image_rgb = _read_image_rgb_uint8(row["path"], image_size=image_size)
        pixels = image_rgb.reshape(-1, 3)

        if len(pixels) > pixels_per_image:
            pixel_indices = rng.choice(len(pixels), size=pixels_per_image, replace=False)
            pixels = pixels[pixel_indices]

        pixel_frame = pd.DataFrame(pixels, columns=["red", "green", "blue"])
        pixel_frame["brightness"] = pixel_frame[["red", "green", "blue"]].mean(axis=1).round(1)
        pixel_frame["class_name"] = row["class_name"]
        sampled_frames.append(pixel_frame)

    if not sampled_frames:
        return pd.DataFrame(columns=["red", "green", "blue", "brightness", "class_name"])

    rgb_points = pd.concat(sampled_frames, ignore_index=True)
    rgb_points["rgb_color"] = rgb_points.apply(
        lambda row: f"rgb({row['red']},{row['green']},{row['blue']})",
        axis=1,
    )
    return rgb_points


def plot_core_color_and_quality(records: pd.DataFrame) -> None:
    rgb_plot = sample_rgb_distribution(records)

    fig_rgb = go.Figure(
        data=[
            go.Scatter3d(
                x=rgb_plot["red"],
                y=rgb_plot["green"],
                z=rgb_plot["blue"],
                mode="markers",
                marker=dict(
                    size=3,
                    opacity=0.5,
                    color=rgb_plot["rgb_color"],
                ),
                customdata=np.stack(
                    [
                        rgb_plot["class_name"],
                        rgb_plot["brightness"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "Class: %{customdata[0]}<br>"
                    "R: %{x}<br>"
                    "G: %{y}<br>"
                    "B: %{z}<br>"
                    "Brightness: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )
    fig_rgb.update_layout(
        title="RGB Color Space Distribution",
        scene=dict(
            xaxis=dict(title="Red", range=[0, 255], backgroundcolor="rgb(245,245,245)"),
            yaxis=dict(title="Green", range=[0, 255], backgroundcolor="rgb(245,245,245)"),
            zaxis=dict(title="Blue", range=[0, 255], backgroundcolor="rgb(245,245,245)"),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig_rgb.show()

    quality_plot = records.sample(min(4000, len(records)), random_state=42).copy()
    fig_quality = px.scatter(
        quality_plot,
        x="sharpness",
        y="contrast",
        color="class_name",
        opacity=0.65,
        title="Image Quality Metrics",
        labels={
            "sharpness": "Sharpness (Laplacian Variance)",
            "contrast": "Contrast (Standard Deviation)",
            "class_name": "Class",
        },
        hover_data={
            "class_name": True,
            "brightness": ":.1f",
            "sharpness": ":.1f",
            "contrast": ":.1f",
            "path": False,
        },
    )
    fig_quality.update_traces(marker=dict(size=8))
    fig_quality.update_layout(margin=dict(l=40, r=20, t=60, b=40))
    fig_quality.show()

    px.histogram(records, x="sharpness", nbins=40, title="Sharpness Distribution").show()


def get_blur_extremes(records: pd.DataFrame, top_k: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = records.sort_values("sharpness")
    return ordered.head(top_k), ordered.tail(top_k)


def plot_blur_extremes(records: pd.DataFrame, top_k: int = 5) -> None:
    blurry, sharp = get_blur_extremes(records, top_k=top_k)
    combined = [("Blurriest", blurry), ("Sharpest", sharp)]

    subplot_titles: List[str] = []
    for label, frame in combined:
        for _, row in frame.iterrows():
            subplot_titles.append(f"{label}<br>{row['class_name']}<br>sharp={row['sharpness']:.1f}")

    fig = make_subplots(
        rows=2,
        cols=top_k,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    for row_idx, (label, frame) in enumerate(combined):
        for col_idx, (_, row) in enumerate(frame.iterrows()):
            img = _read_image_rgb_uint8(row["path"], image_size=256)
            fig.add_trace(go.Image(z=img), row=row_idx + 1, col=col_idx + 1)

    fig.update_layout(
        title="Blurriest vs Sharpest Samples",
        height=700,
        width=260 * top_k,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.show()


def build_classification_summary(records: pd.DataFrame) -> Dict[str, object]:
    class_counts = records.groupby("class_name").size().sort_values(ascending=False)
    min_count = int(class_counts.min())
    max_count = int(class_counts.max())

    return {
        "total_images": int(len(records)),
        "num_classes": int(records["class_name"].nunique()),
        "class_counts": class_counts,
        "max_count": max_count,
        "min_count": min_count,
        "imbalance_ratio": max_count / max(min_count, 1),
    }


def display_classification_overview(summary: Dict[str, object]) -> None:
    print("CLASSIFICATION EDA REPORT")
    print(f"Total images for classification: {summary['total_images']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Max count: {summary['max_count']}")
    print(f"Min count: {summary['min_count']}")
    print(f"Imbalance ratio: {summary['imbalance_ratio']:.2f}x")


def plot_classification_distribution(records: pd.DataFrame) -> None:
    class_counts = records.groupby("class_name").size().sort_values(ascending=False)
    fig1 = px.bar(
        x=class_counts.index,
        y=class_counts.values,
        title="Class Distribution",
        labels={"x": "Class", "y": "Image count"},
    )
    fig1.update_layout(xaxis_tickangle=-45)
    fig1.show()

    imbalance = class_counts.to_frame(name="image_count")
    imbalance["rank"] = np.arange(1, len(imbalance) + 1)
    fig2 = px.line(
        imbalance,
        x="rank",
        y="image_count",
        markers=True,
        title="Class Balance Curve",
        labels={"rank": "Class rank", "image_count": "Image count"},
    )
    fig2.show()


def make_stratified_splits(
    records: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        records,
        train_size=train_size,
        random_state=random_state,
        stratify=records["class_name"],
    )

    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=random_state,
        stratify=temp_df["class_name"],
    )

    split_df = records.copy()
    split_df["split"] = "unassigned"
    split_df.loc[train_df.index, "split"] = "train"
    split_df.loc[val_df.index, "split"] = "val"
    split_df.loc[test_df.index, "split"] = "test"
    return split_df


def plot_split_distribution(split_df: pd.DataFrame) -> None:
    overall = split_df["split"].value_counts().reindex(["train", "val", "test"])
    fig1 = px.bar(
        x=overall.index,
        y=overall.values,
        title="Overall Split Distribution",
        labels={"x": "Split", "y": "Image count"},
    )
    fig1.show()

    top_classes = split_df["class_name"].value_counts().head(10).index
    class_counts = (
        split_df.loc[split_df["class_name"].isin(top_classes)]
        .groupby(["class_name", "split"])
        .size()
        .reset_index(name="count")
    )
    fig2 = px.bar(
        class_counts,
        x="class_name",
        y="count",
        color="split",
        barmode="group",
        title="Top Class Split Distribution",
        labels={"class_name": "Class", "count": "Image count"},
    )
    fig2.update_layout(xaxis_tickangle=-45)
    fig2.show()


def sample_records_for_features(
    records: pd.DataFrame,
    sample_per_class: int,
    max_images: int,
    random_state: int,
) -> pd.DataFrame:
    sampled_frames = []
    for _, group in records.groupby("class_name"):
        sampled_frames.append(
            group.sample(
                n=min(sample_per_class, len(group)),
                random_state=random_state,
            )
        )

    sampled = pd.concat(sampled_frames, ignore_index=False).sample(
        frac=1.0,
        random_state=random_state,
    )
    return sampled.head(max_images).reset_index(drop=True)


def _extract_resnet50_features_torchvision(
    image_paths: Sequence[str],
    image_size: int = 224,
    batch_size: int = 16,
) -> np.ndarray:
    try:
        import torch
        from torchvision import models, transforms
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "ResNet50 feature extraction requires torch and torchvision. "
            "Please install them before running the classification feature section."
        ) from exc

    device = torch.device("cuda")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start_idx in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start_idx : start_idx + batch_size]
            batch_tensors = []
            for path in batch_paths:
                with Image.open(path) as img:
                    batch_tensors.append(transform(img.convert("RGB")))
            batch = torch.stack(batch_tensors).to(device, non_blocking=device.type == "cuda")
            feats = model(batch).detach().cpu().numpy()
            outputs.append(feats)

    return np.vstack(outputs)


def extract_image_features(
    feature_records: pd.DataFrame,
    config: FeatureExtractionConfig,
) -> Tuple[np.ndarray, str]:
    image_paths = feature_records["path"].tolist()
    features = _extract_resnet50_features_torchvision(
        image_paths=image_paths,
        image_size=config.image_size,
        batch_size=config.batch_size,
    )
    return features, "resnet50_imagenet"


def plot_feature_embeddings(
    features: np.ndarray,
    labels: Sequence[str],
    use_umap: bool = True,
    random_state: int = 42,
) -> None:
    pca_2d = PCA(n_components=2, random_state=random_state)
    pca_points = pca_2d.fit_transform(features)

    perplexity = min(30, max(2, len(features) // 4), len(features) - 1)
    tsne = TSNE(n_components=2, random_state=random_state, init="pca", perplexity=perplexity)
    tsne_points = tsne.fit_transform(features)

    emb_df = pd.DataFrame(
        {
            "pca_x": pca_points[:, 0],
            "pca_y": pca_points[:, 1],
            "tsne_x": tsne_points[:, 0],
            "tsne_y": tsne_points[:, 1],
            "label": list(labels),
        }
    )

    px.scatter(emb_df, x="pca_x", y="pca_y", color="label", title="PCA 2D Projection").show()
    px.scatter(emb_df, x="tsne_x", y="tsne_y", color="label", title="t-SNE 2D Projection").show()

    if use_umap:
        try:
            import umap

            reducer = umap.UMAP(random_state=random_state)
            umap_points = reducer.fit_transform(features)
            umap_df = pd.DataFrame({"umap_x": umap_points[:, 0], "umap_y": umap_points[:, 1], "label": list(labels)})
            px.scatter(umap_df, x="umap_x", y="umap_y", color="label", title="UMAP 2D Projection").show()
        except Exception as exc:  # pragma: no cover
            display(f"UMAP unavailable: {exc}")

    pca_full = PCA().fit(features)
    ev_df = pd.DataFrame(
        {
            "components": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
            "cum_explained_variance": np.cumsum(pca_full.explained_variance_ratio_),
        }
    )
    px.line(
        ev_df,
        x="components",
        y="cum_explained_variance",
        markers=True,
        title="PCA Explained Variance",
        labels={"components": "Number of components", "cum_explained_variance": "Cumulative explained variance"},
    ).show()


def build_similarity_outputs(
    features: np.ndarray,
    feature_records: pd.DataFrame,
    n_clusters: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_centroids = (
        pd.DataFrame(features)
        .assign(class_name=feature_records["class_name"].values)
        .groupby("class_name")
        .mean()
    )

    sim = cosine_similarity(class_centroids.values)
    sim_df = pd.DataFrame(sim, index=class_centroids.index, columns=class_centroids.index)

    pairs: List[Tuple[str, str, float]] = []
    for i, row_name in enumerate(sim_df.index):
        for j, col_name in enumerate(sim_df.columns):
            if j <= i:
                continue
            pairs.append((row_name, col_name, float(sim_df.iloc[i, j])))

    top_pairs = (
        pd.DataFrame(pairs, columns=["class_1", "class_2", "similarity"])
        .sort_values("similarity", ascending=False)
        .reset_index(drop=True)
    )

    cluster_count = min(n_clusters, len(class_centroids))
    clustering = AgglomerativeClustering(n_clusters=cluster_count)
    cluster_labels = clustering.fit_predict(class_centroids.values)
    cluster_df = pd.DataFrame(
        {
            "class_name": class_centroids.index,
            "cluster": cluster_labels + 1,
        }
    ).sort_values(["cluster", "class_name"])

    return sim_df, top_pairs, cluster_df


def plot_similarity_outputs(
    sim_df: pd.DataFrame,
    top_pairs: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> None:
    fig = px.imshow(
        sim_df.values,
        x=sim_df.columns.tolist(),
        y=sim_df.index.tolist(),
        color_continuous_scale="Viridis",
        title="Class Similarity Matrix",
        aspect="auto",
    )
    fig.show()

    display(top_pairs.head(10))
    display(cluster_df.groupby("cluster")["class_name"].apply(list).reset_index(name="classes"))
