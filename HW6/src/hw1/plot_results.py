"""
讀取 run_llm_eval.py 的輸出，產生報告用圖表（PNG）。

  fig1_strategy_precision.png  精度 × 策略 overall 分組長條圖
  fig2_category_heatmap.png    （精度/策略）× 類別 熱力圖
  fig3_vram.png                各精度 權重大小 / 推論峰值 VRAM 長條圖
  fig4_quality_vs_vram.png     品質-顯存 取捨散佈圖（越左上越好）

用法：python src/hw1/plot_results.py --out-dir data/hw1/results/final
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import DATA, FONT  # noqa: E402

if os.path.exists(FONT):  # 中文字型（由 src/hw2/make_synthetic_images.py 下載）
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()

PREC_ORDER = ["bf16", "fp16", "int8", "int4"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def ordered(values):
    return [p for p in PREC_ORDER if p in set(values)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(DATA, "hw1", "results", "final"))
    a = ap.parse_args()
    d = a.out_dir
    judged = pd.read_csv(os.path.join(d, "judged.csv"))
    vram = pd.read_csv(os.path.join(d, "vram.csv"))
    precs = ordered(judged["precision"].unique())
    strategies = list(dict.fromkeys(judged["strategy"]))

    # 圖 1：分組長條圖 + 標準誤
    g = judged.groupby(["precision", "strategy"])["overall"]
    mean, sem = g.mean(), g.sem()
    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(strategies)
    x = np.arange(len(precs))
    for i, s in enumerate(strategies):
        m = [mean.get((p, s), np.nan) for p in precs]
        e = [sem.get((p, s), 0) for p in precs]
        bars = ax.bar(x + i * w - 0.4 + w / 2, m, w, yerr=e, capsize=3, label=s, color=COLORS[i % 4])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, precs)
    ax.set_ylim(0, 10)
    ax.set_ylabel("LLM-Judge overall（1–10）")
    ax.set_title("不同精度 × Prompt 策略的平均分數")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig1_strategy_precision.png"), dpi=150)

    # 圖 2：類別熱力圖
    pv = judged.pivot_table(index=["precision", "strategy"], columns="category", values="overall", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.4 * pv.shape[1] + 2, 0.45 * pv.shape[0] + 1.5))
    im = ax.imshow(pv.values, cmap="RdYlGn", vmin=1, vmax=10, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), [f"{p}/{s}" for p, s in pv.index])
    for (r, c), v in np.ndenumerate(pv.values):
        ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    ax.set_title("各能力類別分數")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig2_category_heatmap.png"), dpi=150)

    # 圖 3：VRAM（每個精度取各策略的最大峰值）
    v = vram.groupby("precision").agg(weights=("load_weights_footprint_mib", "first"),
                                      torch_peak=("infer_torch_peak_alloc_mib", "max"),
                                      nvml_peak=("infer_nvml_peak_used_mib", "max"),
                                      nvml_avg=("infer_nvml_avg_used_mib", "mean")).reindex(precs) / 1024
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(precs))
    for i, col in enumerate(["weights", "torch_peak", "nvml_avg", "nvml_peak"]):
        bars = ax.bar(x + i * 0.2 - 0.3, v[col], 0.2, label=col, color=COLORS[i])
        ax.bar_label(bars, fmt="%.1f", fontsize=7)
    ax.set_xticks(x, precs)
    ax.set_ylabel("GiB")
    ax.set_title("各精度 VRAM 使用量")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig3_vram.png"), dpi=150)

    # 圖 4：品質 vs VRAM 取捨
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, s in enumerate(strategies):
        xs = [v.loc[p, "nvml_peak"] for p in precs]
        ys = [mean.get((p, s), np.nan) for p in precs]
        ax.plot(xs, ys, "o-", label=s, color=COLORS[i % 4])
        for p, xx, yy in zip(precs, xs, ys):
            ax.annotate(p, (xx, yy), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("推論峰值 VRAM（GiB, NVML）")
    ax.set_ylabel("overall 分數")
    ax.set_title("品質 vs 顯存取捨")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig4_quality_vs_vram.png"), dpi=150)
    print("圖表已輸出至", d)


if __name__ == "__main__":
    main()
