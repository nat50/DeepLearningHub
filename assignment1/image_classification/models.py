from typing import Dict, List

import timm
import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    A unified head shared by all backbones so the experiment pipeline differs
    only in the feature extractor.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ImageClassifier(nn.Module):
    """
    image -> pretrained backbone -> shared head -> logits
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet50":
            self.backbone = timm.create_model(
                "resnetv2_50x1_bit.goog_in21k",
                pretrained=pretrained,
                num_classes=0,
            )
        elif backbone == "mobilenet_v3":
            self.backbone = timm.create_model(
                "mobilenetv3_large_100.miil_in21k",
                pretrained=pretrained,
                num_classes=0,
            )
        elif backbone == "vit_b16":
            self.backbone = timm.create_model(
                "vit_base_patch16_224.orig_in21k",
                pretrained=pretrained,
                num_classes=0,
            )
        elif backbone == "swin_b":
            self.backbone = timm.create_model(
                "swin_base_patch4_window7_224.ms_in22k",
                pretrained=pretrained,
                num_classes=0,
            )
        else:
            raise ValueError(
                "Unsupported backbone. Available: resnet50, mobilenet_v3, vit_b16, swin_b"
            )

        if device is not None:
            self.backbone = self.backbone.to(device)

        self.feature_dim = self._infer_feature_dim(device=device)
        self.head = ClassificationHead(in_features=self.feature_dim, num_classes=num_classes)
        if device is not None:
            self.head = self.head.to(device)
        self.freeze_backbone()

    def _infer_feature_dim(self, device: torch.device | None = None) -> int:
        try:
            backbone_device = next(self.backbone.parameters()).device
        except StopIteration:
            backbone_device = torch.device("cpu")

        input_size = getattr(self.backbone, "default_cfg", {}).get("input_size", (3, 224, 224))
        if len(input_size) == 4:
            input_size = input_size[1:]

        dummy = torch.zeros(1, *input_size, device=device or backbone_device)
        training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(dummy)
        if training:
            self.backbone.train()
        return int(features.shape[-1])

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)


def create_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> ImageClassifier:
    return ImageClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
    )


def list_backbones() -> List[str]:
    return ["resnet50", "mobilenet_v3", "vit_b16", "swin_b"]


def get_backbone_config(backbone: str) -> Dict[str, str]:
    if backbone == "resnet50":
        return {
            "key": "resnet50",
            "timm_name": "resnetv2_50x1_bit.goog_in21k",
            "family": "cnn",
            "pretrained_source": "BiT / ImageNet-21k",
        }
    if backbone == "mobilenet_v3":
        return {
            "key": "mobilenet_v3",
            "timm_name": "mobilenetv3_large_100.miil_in21k",
            "family": "cnn",
            "pretrained_source": "MIIL / ImageNet-21k",
        }
    if backbone == "vit_b16":
        return {
            "key": "vit_b16",
            "timm_name": "vit_base_patch16_224.orig_in21k",
            "family": "transformer",
            "pretrained_source": "ImageNet-21k",
        }
    if backbone == "swin_b":
        return {
            "key": "swin_b",
            "timm_name": "swin_base_patch4_window7_224.ms_in22k",
            "family": "transformer",
            "pretrained_source": "ImageNet-22k",
        }
    raise ValueError(
        "Unsupported backbone. Available: resnet50, mobilenet_v3, vit_b16, swin_b"
    )


if __name__ == "__main__":
    device = torch.device("cuda")
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    for backbone_name in list_backbones():
        model = create_model(
            backbone=backbone_name,
            num_classes=101,
            pretrained=False,
            device=device,
        ).to(device)
        logits = model(dummy_input)
        print(
            f"{backbone_name:16s} -> feature_dim={model.feature_dim:4d} "
            f"logits_shape={tuple(logits.shape)}"
        )