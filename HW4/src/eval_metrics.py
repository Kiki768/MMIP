"""評估指標與繪圖。

Quiz 2 / 3：分類指標（Accuracy、Precision、Recall、F1、AUC / Macro-AUC）、混淆矩陣、ROC、模型比較圖。
Quiz 4 的 BLEU / BERTScore / Gemini 之後也加在這個檔案。
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def classification_metrics(y_true, y_pred, probs):
    """二分類與多分類皆可。

    Precision / Recall / F1 使用 macro 平均（各類別平均，不受類別大小影響）。
    AUC：二分類回報 "AUC"；多分類回報 "Macro-AUC"（one-vs-rest，各類別 AUC 的平均）。
    """
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    probs = np.asarray(probs)
    if probs.shape[1] == 2:
        auc_name, auc = "AUC", roc_auc_score(y_true, probs[:, 1])
    else:
        auc_name, auc = "Macro-AUC", roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": p,
        "Recall": r,
        "F1": f1,
        auc_name: auc,
    }


def per_class_auc(y_true, probs, class_names):
    """每個類別的 one-vs-rest AUC，回傳 {類別名稱: AUC}；Macro-AUC 就是這些值的平均。"""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    return {name: roc_auc_score(y_true == c, probs[:, c]) for c, name in enumerate(class_names)}


def print_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))


def plot_history(histories, save_path=None):
    """histories: {模型名稱: history}。左圖為 loss，右圖為 accuracy (train & val)。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for name, h in histories.items():
        epochs = range(1, len(h["train_loss"]) + 1)
        
        # --- 左圖：Loss (Train vs Val) ---
        line_loss, = axes[0].plot(epochs, h["train_loss"], label=f"{name} train")
        axes[0].plot(epochs, h["val_loss"], "--", color=line_loss.get_color(), label=f"{name} val")
        
        # --- 右圖：Accuracy (Train vs Val) ---
        line_acc, = axes[1].plot(epochs, h["train_acc"], label=f"{name} train")
        axes[1].plot(epochs, h["val_acc"], "--", marker="o", color=line_acc.get_color(), label=f"{name} val")

    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
        
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    k = len(class_names)
    side = max(4.5, 0.65 * k)  # 類別多時放大圖片
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    ax.imshow(cm, cmap="Blues")
    ax.set(
        title=title,
        xlabel="Predicted",
        ylabel="True",
        xticks=range(k),
        yticks=range(k),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    if k > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    fs = 10 if k <= 5 else 8
    for i in range(k):
        for j in range(k):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=fs,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(y_true, probs, class_names, title="ROC curves (one-vs-rest)", save_path=None):
    """每個類別各畫一條 one-vs-rest ROC 曲線，圖例標示該類別 AUC。"""
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true == c, probs[:, c])
        ax.plot(fpr, tpr, lw=1.2, label=f"{name} ({roc_auc_score(y_true == c, probs[:, c]):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title=title, xlabel="False positive rate", ylabel="True positive rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def show_image_predictions(dataset, y_true, y_pred, probs, class_names, indices, title="",
                           save_path=None, mean=IMAGENET_MEAN, std=IMAGENET_STD, ncols=6):
    """顯示 dataset 中指定 indices 的影像與預測結果。

    dataset[i] 需回傳已標準化的 (影像 tensor, 標籤)；標題為 T=真實類別、P=預測類別（信心度），
    預測正確以 ✓ 與綠色標示，錯誤以 ✗ 與紅色標示。
    """
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    n = len(indices)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, indices):
        img, _ = dataset[int(i)]
        img = (img.numpy() * std + mean).clip(0, 1).transpose(1, 2, 0)
        ax.imshow(img)
        ok = y_true[i] == y_pred[i]
        ax.set_title(
            f"{'✓' if ok else '✗'} T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]} ({probs[i].max():.2f})",
            fontsize=8,
            color="green" if ok else "red",
        )
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def compare_models(results):
    """results: {模型名稱: {'metrics': dict, 'params': int, 'train_time': 秒, 'infer_time': 秒}}
    回傳整理好的比較表（DataFrame）。"""
    rows = {}
    for name, r in results.items():
        rows[name] = {
            **r["metrics"],
            "Params": r["params"],
            "Train time (s)": r["train_time"],
            "Inference time (s)": r["infer_time"],
        }
    return pd.DataFrame(rows).T


def plot_metric_bars(df, metrics=("Accuracy", "Precision", "Recall", "F1", "AUC", "Macro-AUC"), save_path=None):
    """把比較表中的指標畫成並排長條圖（表中沒有的指標會自動略過）。"""
    metrics = [m for m in metrics if m in df.columns]
    ax = df[metrics].plot(kind="bar", figsize=(8, 4), rot=0)
    lo = max(0.0, df[metrics].min().min() - 0.05)
    ax.set(title="Model comparison", ylabel="Score", ylim=(lo, 1.0))
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_per_class_auc(df, save_path=None):
    """df：列為類別、欄為模型的每類 AUC 表，畫成並排長條圖。"""
    ax = df.plot(kind="bar", figsize=(9, 4), rot=30)
    lo = max(0.0, df.min().min() - 0.02)
    ax.set(title="Per-class AUC (one-vs-rest)", ylabel="AUC", ylim=(lo, 1.0))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
