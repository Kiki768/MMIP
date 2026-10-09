"""Quiz 3：預訓練的 ViT 與 ResNet（torchvision），換上 10 類分類頭後微調。

權重第一次使用時會自動從網路下載（ViT-B/16 約 330MB、ResNet50 約 100MB）。
torchvision 在函式內才匯入，因此只做文字任務（Quiz 2）時不需要安裝 torchvision。
"""
import torch.nn as nn


def _freeze_all_but(model, trainable_prefix):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(trainable_prefix)


def build_vit(num_classes=10, pretrained=True, freeze_backbone=False):
    """ViT-B/16（ImageNet-1k 預訓練，輸入 224x224）。換掉最後的分類頭 heads.head。"""
    from torchvision import models

    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    if freeze_backbone:  # 只訓練分類頭（linear probing）
        _freeze_all_but(model, "heads")
    return model


def build_resnet(num_classes=10, pretrained=True, arch="resnet50", freeze_backbone=False):
    """ResNet（預設 ResNet50，ImageNet-1k 預訓練）。換掉最後的全連接層 fc。"""
    from torchvision import models

    table = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    }
    if arch not in table:
        raise ValueError(f"arch 需為 {list(table)} 之一")
    ctor, w = table[arch]
    model = ctor(weights=w if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_backbone:
        _freeze_all_but(model, "fc")
    return model
