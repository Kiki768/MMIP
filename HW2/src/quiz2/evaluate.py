"""Step 5：預測、評估指標、畫圖"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay)


@torch.no_grad()
def predict_proba(model, X, device="cpu"): # 輸出預測違約的機率
    model.eval()
    logits = model(torch.from_numpy(X).to(device))
    return torch.sigmoid(logits).cpu().numpy()


def compute_metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def predict_one_sample(model, X_val, y_val, idx=0, device="cpu"): 
    """展示預測 Validation Dataset 的一筆 sample"""
    prob = float(predict_proba(model, X_val[idx:idx + 1], device)[0])
    return {
        "index": idx,
        "prob_default": round(prob, 4),
        "predicted": int(prob >= 0.5),
        "actual": int(y_val[idx]),
    }


def plot_loss(history, title, path): # 單一模型的 loss 曲線
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.show()

# 進階：模型調參的前後比較
def plot_compare(h_base, h_imp, path): # 兩個模型的 loss 比較
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, name in [(axes[0], "train_loss", "Training Loss"),
                          (axes[1], "val_loss", "Validation Loss")]:
        ax.plot(h_base[key], label="Baseline")
        ax.plot(h_imp[key], label="Improved")
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.show()


def plot_metrics_bar(m_base, m_imp, path): # 指標比較
    keys = list(m_base.keys())
    x = np.arange(len(keys)); w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w / 2, [m_base[k] for k in keys], w, label="Baseline")
    plt.bar(x + w / 2, [m_imp[k] for k in keys], w, label="Improved")
    plt.xticks(x, keys); plt.ylim(0, 1); plt.legend(); plt.title("Validation Metrics")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.show()

# 畫混淆矩陣
def plot_confusion_matrix(y_true, prob, title, path, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Default (0)", "Default (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.show()
    return cm