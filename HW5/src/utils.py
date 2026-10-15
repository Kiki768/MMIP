# -*- coding: utf-8 -*-
"""共用工具：日誌、隨機種子、裝置選擇、骨幹模型建立、指標計算、畫圖與表格輸出。

注意：圖片內的標題一律使用英文，避免 Matplotlib 預設字型缺少中文字形而顯示成方框。
"""
import logging
import random
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# 基礎工具
# --------------------------------------------------------------------------- #
def get_logger(name: str = "mmdl") -> logging.Logger:
    """取得統一格式的 logger（重複呼叫不會重複加 handler）。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


logger = get_logger()


def ensure_dir(path: Union[str, Path]) -> Path:
    """若資料夾不存在則（遞迴）自動建立，並回傳 Path 物件。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """固定 Python / NumPy / PyTorch 的隨機種子，讓實驗可重現。"""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto"):
    """解析使用者指定的裝置；指定的裝置不可用時自動退回 CPU 並提示。"""
    import torch

    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("指定了 %s 但找不到可用的 CUDA，自動改用 CPU。", requested)
        return torch.device("cpu")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        logger.warning("指定了 mps 但不可用，自動改用 CPU。")
        return torch.device("cpu")
    return torch.device(requested)


def save_table(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    """將結果表格存成 CSV（utf-8-sig，Excel 直接開啟不會亂碼），並在終端機列印。"""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("結果表格已儲存：%s", path)
    try:  # 在 Notebook 內以表格形式顯示
        from IPython.display import display

        display(df)
    except Exception:
        print(df.to_string(index=False))
    return path


def count_parameters(model, trainable_only: bool = False) -> int:
    """計算模型參數量。"""
    return sum(
        p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)
    )


# --------------------------------------------------------------------------- #
# 影像骨幹（Quiz 1、Quiz 2 共用）
# --------------------------------------------------------------------------- #
def build_image_backbone(name: str) -> Tuple["torch.nn.Module", int]:  # noqa: F821
    """建立 ImageNet 預訓練影像骨幹，移除分類頭後輸出特徵向量。

    Args:
        name: "resnet50"（輸出 2048 維）或 "vit_b_16"（輸出 768 維）。
    Returns:
        (model, feature_dim)
    """
    import torch.nn as nn
    from torchvision import models

    name = name.lower()
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()  # 去掉 1000 類分類層，保留 global-avg-pool 後的 2048 維
        return model, 2048
    if name in ("vit_b_16", "vit-b/16", "vit"):
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = nn.Identity()  # 保留 [CLS] token 經 encoder 後的 768 維
        return model, 768
    raise ValueError(f"不支援的影像骨幹：{name}（可用：resnet50, vit_b_16）")


def build_imagenet_transform(size: int = 224):
    """ImageNet 標準前處理（ViT-B/16 需固定 224x224 輸入）。"""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# 指標計算
# --------------------------------------------------------------------------- #
def topk_retrieval_metrics(
    sim: "torch.Tensor",  # noqa: F821
    query_labels: "torch.Tensor",  # noqa: F821
    gallery_labels: "torch.Tensor",  # noqa: F821
    k: int = 5,
) -> Tuple[Dict[str, float], "torch.Tensor", "torch.Tensor"]:  # noqa: F821
    """依相似度矩陣計算檢索指標。

    - Top-1 Acc：相似度最高的 1 張與 Query 同類別的比例
    - Top-5 Acc：前 k 張中「至少一張」與 Query 同類別的比例（hit rate）
    - Precision@5：前 k 張中同類別的平均比例

    Returns:
        (metrics_dict, topk_indices [Q,k], topk_scores [Q,k])
    """
    scores, idx = sim.topk(k, dim=1)
    retrieved = gallery_labels[idx]  # [Q, k]
    match = retrieved == query_labels.unsqueeze(1)
    metrics = {
        "top1": match[:, 0].float().mean().item() * 100,
        f"top{k}": match.any(dim=1).float().mean().item() * 100,
        f"precision@{k}": match.float().mean().item() * 100,
    }
    return metrics, idx, scores


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_prob: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    """二元分類指標：Accuracy、F1（正類=1）、Macro-F1，若提供機率則加算 AUROC。"""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    out = {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "auroc": float("nan"),
    }
    if y_prob is not None:
        try:
            out["auroc"] = roc_auc_score(y_true, y_prob) * 100
        except ValueError:  # 只有單一類別時無法計算
            pass
    return out


# --------------------------------------------------------------------------- #
# 畫圖
# --------------------------------------------------------------------------- #
def _frame(ax, color: str, width: float = 3.0) -> None:
    """隱藏刻度並以彩色外框標示（綠=同類別、紅=不同類別、藍=Query）。"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(width)


def plot_retrieval_samples(rows: List[dict], class_names: Sequence[str], save_path) -> Path:
    """拼接繪製「Query + Top-K 檢索結果」。

    rows 中每個元素：
        {"model": str, "query_img": HxWx3 uint8, "query_label": int,
         "retrieved": [(img, label, score), ...]}
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    n_rows = len(rows)
    n_cols = 1 + max(len(r["retrieved"]) for r in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.35 * n_rows), squeeze=False)
    for r, row in enumerate(rows):
        ax = axes[r][0]
        ax.imshow(row["query_img"])
        ax.set_title(f"[{row['model']}]\nQuery: {class_names[row['query_label']]}", fontsize=8)
        _frame(ax, "tab:blue")
        for c, (img, label, score) in enumerate(row["retrieved"], start=1):
            ax = axes[r][c]
            ax.imshow(img)
            ok = label == row["query_label"]
            ax.set_title(f"{class_names[label]}\ncos={score:.3f}", fontsize=8)
            _frame(ax, "tab:green" if ok else "tab:red")
        for c in range(1 + len(row["retrieved"]), n_cols):
            axes[r][c].axis("off")
    fig.suptitle("Image Retrieval: Query (blue) and Top-5 results (green=correct, red=wrong)", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()  # 在 Notebook 內直接顯示（上一行已同步存檔）
    logger.info("檢索視覺化已儲存：%s", save_path)
    return save_path


def plot_training_curves(histories: Dict[str, pd.DataFrame], save_path) -> Path:
    """繪製各模型組合的訓練 / 驗證 loss 與驗證 F1 曲線。"""
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ax1.plot(h["epoch"], h["train_loss"], marker="o", label=f"{name} train")
        ax1.plot(h["epoch"], h["val_loss"], marker="x", linestyle="--", label=f"{name} val")
        ax2.plot(h["epoch"], h["val_f1"], marker="o", label=name)
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.legend(fontsize=7)
    ax2.set_title("Validation F1 (%)")
    ax2.set_xlabel("epoch")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()  # 在 Notebook 內直接顯示（上一行已同步存檔）
    logger.info("訓練曲線已儲存：%s", save_path)
    return save_path


def plot_caption_samples(samples: List[dict], save_path, ncols: int = 3) -> Path:
    """繪製「影像 + 生成描述 + 其中一句 Ground Truth」。

    samples 中每個元素：{"image": PIL.Image/ndarray, "generated": str, "reference": str}
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    n = len(samples)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.8 * nrows), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        ax.axis("off")
        if i >= n:
            continue
        s = samples[i]
        ax.imshow(s["image"])
        gen = "\n".join(textwrap.wrap("Generated: " + s["generated"], 48))
        ref = "\n".join(textwrap.wrap("GT: " + s["reference"], 48))
        ax.set_title(f"{gen}\n{ref}", fontsize=9, loc="left")
    fig.suptitle("CLIP + GPT-2 Image Captioning samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()  # 在 Notebook 內直接顯示（上一行已同步存檔）
    logger.info("Caption 範例已儲存：%s", save_path)
    return save_path
