import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    return cm, metrics

def compare_models(results_dict):
    # results_dict = {'model1': metrics_dict, 'model2': metrics_dict}
    return pd.DataFrame(results_dict).T

# 把數據畫成表格圖
def save_table_as_image(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 計算適合表格的圖表寬高
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(
        figsize=(max(6, n_cols * 1.8), max(2.5, (n_rows + 1) * 0.5))
    )
    ax.axis("off")  # 隱藏外圍座標軸

    # 繪製表格
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    # 美化樣式（字體大小、單元格高度、標題列背景色）
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # 設定標頭樣式（類似 Word 的深色標題橫列）
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == -1:  # 欄位名稱或索引
            cell.set_facecolor("#2F5597")  # 經典 Word 深藍色
            cell.set_text_props(color="white", weight="bold")
        else:
            if row % 2 == 1:
                cell.set_facecolor("#F2F2F2")  # 斑馬紋交替淺灰底色

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"表格圖片已儲存至: {save_path}")