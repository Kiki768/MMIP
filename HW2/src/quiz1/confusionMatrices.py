import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrices(
    cms, thresholds, labels=("No Diabetes", "Diabetes"), save_path=None
):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, cm, t in zip(axes, cms, thresholds):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Threshold = {t}")

    plt.tight_layout()

    # 儲存圖片（必須在 plt.show() 之前執行）
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"混淆矩陣圖片已儲存至: {save_path}")

    plt.show()