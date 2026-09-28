"""Quiz 3：ROC Curve 與 AUC"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc_curves(y_true, probs, title, path):
    """probs 是 dict：{"模型名稱": 預測機率}，可以放一個或多個模型"""
    plt.figure(figsize=(7, 6))
    aucs = {}
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        aucs[name] = auc
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess (AUC = 0.5)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.xlim(0, 1); plt.ylim(0, 1.02); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.show()
    return aucs