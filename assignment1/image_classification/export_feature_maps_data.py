from __future__ import annotations

import base64
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import create_model


ROOT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT_DIR / "results" / "food101_classification"
DATA_ROOT = ROOT_DIR / "data" / "food-101"
IMAGE_ROOT = DATA_ROOT / "images"
META_TEST = DATA_ROOT / "meta" / "test.txt"
OUTPUT_JSON = ROOT_DIR / "classification" / "data" / "feature_maps_data.json"

SAMPLE_COUNT = 6
FILTERS_PER_LAYER = 16
RANDOM_SEED = 42

MODEL_LAYER_SPECS = {
    "resnet50_no_aug": [
        {"key": "conv_stem", "label": "Conv Stem", "module": "stem"},
        {"key": "stage2", "label": "Stage 2", "module": "stages.1.blocks"},
        {"key": "stage3", "label": "Stage 3", "module": "stages.2.blocks"},
        {"key": "stage4", "label": "Stage 4", "module": "stages.3.blocks"},
    ],
    "mobilenet_v3_no_aug": [
        {"key": "conv_stem", "label": "Conv Stem", "module": "conv_stem"},
        {"key": "mbconv_early", "label": "MBConv Block 2", "module": "blocks.2"},
        {"key": "mbconv_mid", "label": "MBConv Block 3", "module": "blocks.3"},
        {"key": "mbconv_late", "label": "MBConv Block 5", "module": "blocks.5"},
    ],
    "vit_b16_no_aug": [
        {"key": "patch_embed", "label": "Patch Embedding", "module": "patch_embed"},
        {"key": "encoder6", "label": "Encoder Block 6", "module": "blocks.5"},
        {"key": "encoder9", "label": "Encoder Block 9", "module": "blocks.8"},
        {"key": "encoder12", "label": "Encoder Block 12", "module": "blocks.11"},
    ],
    "swin_b_no_aug": [
        {"key": "patch_embed", "label": "Patch Embedding", "module": "patch_embed"},
        {"key": "stage2", "label": "Stage 2", "module": "layers.1.blocks"},
        {"key": "stage3", "label": "Stage 3", "module": "layers.2.blocks"},
        {"key": "stage4", "label": "Stage 4", "module": "layers.3.blocks"},
    ],
}


def _format_food_label(class_name: str) -> str:
    return " ".join(part.capitalize() for part in class_name.split("_"))


def _get_module_by_name(root_module: torch.nn.Module, module_name: str) -> torch.nn.Module:
    module: torch.nn.Module = root_module
    for part in module_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def _resolve_tensor_from_hook_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.detach().cpu()
    if isinstance(output, (list, tuple)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first.detach().cpu()
    raise ValueError("Unsupported hook output type for activation export")


def _to_spatial_feature_map(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert various activation layouts to [C, H, W].
    Supports CNN [B,C,H,W], NHWC, and token outputs [B,N,C].
    """
    tensor = tensor.detach().cpu().float()

    if tensor.ndim == 4:
        # [B, C, H, W] or [B, H, W, C]
        if tensor.shape[0] > 1:
            tensor = tensor[:1]
        tensor = tensor[0]
        d0, d1, d2 = tensor.shape
        if d0 == d1 and d2 != d0:
            # [H, W, C] -> [C, H, W]
            tensor = tensor.permute(2, 0, 1)
        elif not (d1 == d2 and d0 != d1):
            # Fallback: if last dim looks like channels, permute.
            if d2 > d0 and d2 > d1:
                tensor = tensor.permute(2, 0, 1)
        return tensor.contiguous()

    if tensor.ndim == 3:
        # [B, N, C] or [B, C, N]
        if tensor.shape[0] > 1:
            tensor = tensor[:1]
        token_tensor = tensor[0]
        if token_tensor.ndim != 2:
            raise ValueError(f"Unexpected 3D tensor after batch select: {token_tensor.shape}")
        n, m = token_tensor.shape
        if n <= m:
            tokens = token_tensor
        else:
            tokens = token_tensor.t()

        token_count, channels = tokens.shape
        if token_count > 1:
            side = int(math.sqrt(token_count - 1))
            if side * side == token_count - 1:
                tokens = tokens[1:]
                token_count -= 1

        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(f"Cannot reshape token sequence into square map: {token_count}")

        maps = tokens.reshape(side, side, channels).permute(2, 0, 1)
        return maps.contiguous()

    if tensor.ndim == 2:
        # [N, C] tokens (without explicit batch).
        n, m = tensor.shape
        if n <= m:
            tokens = tensor
        else:
            tokens = tensor.t()

        token_count, channels = tokens.shape
        if token_count > 1:
            side = int(math.sqrt(token_count - 1))
            if side * side == token_count - 1:
                tokens = tokens[1:]
                token_count -= 1

        side = int(math.sqrt(token_count))
        if side * side != token_count:
            raise ValueError(f"Cannot reshape 2D token tensor into square map: {token_count}")

        maps = tokens.reshape(side, side, channels).permute(2, 0, 1)
        return maps.contiguous()

    raise ValueError(f"Unsupported activation ndim={tensor.ndim}")


def _normalize_map_to_uint8(feature_map: torch.Tensor) -> np.ndarray:
    array = feature_map.detach().cpu().numpy().astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _select_filter_indices(spatial_maps: torch.Tensor, top_k: int) -> list[int]:
    channels = int(spatial_maps.shape[0])
    if channels <= top_k:
        return list(range(channels))

    flattened = spatial_maps.reshape(channels, -1)
    variances = torch.var(flattened, dim=1, unbiased=False)
    top_indices = torch.topk(variances, k=top_k, largest=True).indices.tolist()
    return sorted(int(idx) for idx in top_indices)


def _collect_sample_paths(sample_count: int) -> list[dict[str, str]]:
    lines = [line.strip() for line in META_TEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(RANDOM_SEED)
    chosen = sorted(rng.sample(lines, k=min(sample_count, len(lines))))

    samples: list[dict[str, str]] = []
    for idx, value in enumerate(chosen, start=1):
        class_name, image_id = value.split("/", 1)
        image_path = IMAGE_ROOT / class_name / f"{image_id}.jpg"
        rel = image_path.relative_to(ROOT_DIR).as_posix()
        samples.append(
            {
                "id": f"sample_{idx}",
                "class_name": class_name,
                "label": f"{_format_food_label(class_name)} - {image_id}",
                "image_path": f"../{rel}",
                "absolute_path": str(image_path),
            }
        )

    return samples


def _load_transform_from_summary(summary: dict[str, Any]) -> transforms.Compose:
    data_cfg = summary.get("data_config", {})
    mean = data_cfg.get("mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("std", [0.229, 0.224, 0.225])

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _run_single_model_export(
    slug: str,
    samples: list[dict[str, str]],
    device: torch.device,
) -> tuple[str, dict[str, Any]]:
    exp_dir = RESULTS_ROOT / slug
    summary_path = exp_dir / "experiment_summary.json"
    checkpoint_path = exp_dir / "best_model.pth"

    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing summary/checkpoint for {slug}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    backbone = summary["backbone"]
    experiment_name = summary["experiment_name"]

    layer_specs = MODEL_LAYER_SPECS[slug]

    model = create_model(
        backbone=backbone,
        num_classes=len(summary.get("class_names", [])) or 101,
        pretrained=False,
        device=device,
    )
    raw_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw_state, dict) and "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        raw_state = raw_state["state_dict"]

    state = {
        key: value
        for key, value in raw_state.items()
        if not (key.endswith("total_ops") or key.endswith("total_params") or ".total_ops" in key or ".total_params" in key)
    }

    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing keys while loading checkpoint for {slug}: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Warning: ignored unexpected keys for {slug}: {unexpected_keys[:5]}")
    model.eval()

    preprocess = _load_transform_from_summary(summary)

    captured: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def _build_hook(layer_key: str):
        def _hook(_module, _inputs, output):
            captured[layer_key] = _resolve_tensor_from_hook_output(output)

        return _hook

    for layer in layer_specs:
        module = _get_module_by_name(model.backbone, layer["module"])
        hooks.append(module.register_forward_hook(_build_hook(layer["key"])))

    model_payload: dict[str, Any] = {
        "slug": slug,
        "backbone": backbone,
        "display_name": experiment_name.split(" (")[0],
        "layers": [
            {
                "key": layer["key"],
                "label": layer["label"],
                "module_name": layer["module"],
            }
            for layer in layer_specs
        ],
        "activations": {},
    }

    with torch.no_grad():
        for sample in samples:
            image_path = Path(sample["absolute_path"])
            with Image.open(image_path) as image:
                image = image.convert("RGB")
            tensor = preprocess(image).unsqueeze(0).to(device)

            captured.clear()
            _ = model(tensor)

            sample_payload: dict[str, Any] = {}
            for layer in layer_specs:
                layer_key = layer["key"]
                if layer_key not in captured:
                    continue

                raw_activation = captured[layer_key]
                spatial_maps = _to_spatial_feature_map(raw_activation)
                filter_indices = _select_filter_indices(spatial_maps, top_k=FILTERS_PER_LAYER)

                selected_filters: list[dict[str, Any]] = []
                for filter_id in filter_indices:
                    map_uint8 = _normalize_map_to_uint8(spatial_maps[filter_id])
                    height, width = map_uint8.shape
                    selected_filters.append(
                        {
                            "filter_id": int(filter_id),
                            "height": int(height),
                            "width": int(width),
                            "data_b64": base64.b64encode(map_uint8.tobytes()).decode("ascii"),
                        }
                    )

                sample_payload[layer_key] = {
                    "shape": [int(dim) for dim in raw_activation.shape],
                    "spatial_shape": [
                        int(spatial_maps.shape[0]),
                        int(spatial_maps.shape[1]),
                        int(spatial_maps.shape[2]),
                    ],
                    "total_filters": int(spatial_maps.shape[0]),
                    "exported_filters": int(len(selected_filters)),
                    "filters": selected_filters,
                }

            model_payload["activations"][sample["id"]] = sample_payload

    for handle in hooks:
        handle.remove()

    return experiment_name, model_payload


def export_feature_maps_data() -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = _collect_sample_paths(SAMPLE_COUNT)

    payload: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filter_scope": "topk_by_variance",
            "filters_per_layer": FILTERS_PER_LAYER,
            "data_encoding": "base64_uint8",
            "sample_count": len(samples),
            "device": str(device),
            "note": "Real feature maps exported from saved NoAug checkpoints.",
        },
        "samples": [
            {
                "id": sample["id"],
                "label": sample["label"],
                "class_name": sample["class_name"],
                "image_path": sample["image_path"],
            }
            for sample in samples
        ],
        "models": {},
    }

    for slug in MODEL_LAYER_SPECS:
        experiment_name, model_payload = _run_single_model_export(slug, samples=samples, device=device)
        payload["models"][experiment_name] = model_payload
        print(f"Exported: {experiment_name} ({slug})")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return OUTPUT_JSON


if __name__ == "__main__":
    path = export_feature_maps_data()
    print(f"Feature map data saved to: {path}")
