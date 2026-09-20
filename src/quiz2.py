import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------
# 0. 讀取影像 + 轉灰階
# ---------------------------------------------------------
def load_image(image_path: str) -> np.ndarray:
    """回傳 RGB 格式的 numpy array (H, W, 3), dtype=uint8"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(
            f"讀不到圖片: {image_path}\n"
            f"請確認檔案路徑是否正確(從執行 main.ipynb 的資料夾為基準)。"
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"已讀取圖片: {image_path}, 尺寸 = {img_rgb.shape}")
    return img_rgb


def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """Histogram Equalization 只能處理單通道影像,先轉成灰階(沿用 Quiz 1 的做法)"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------
# 1. 基礎題:Histogram Equalization (直接用 OpenCV)
# ---------------------------------------------------------
def equalize_opencv(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


# ---------------------------------------------------------
# 2. 進階題:自行用 NumPy 實作 Histogram Equalization
# ---------------------------------------------------------
def equalize_numpy(gray: np.ndarray) -> np.ndarray:
    """
    標準 Histogram Equalization 演算法:
    1. 統計灰階直方圖(0~255,共 256 個 bin)
    2. 計算累積分布函數 CDF
    3. 把 CDF 正規化到 0~255,當作每個灰階值的對應表(lookup table)
    4. 用 lookup table 把原始灰階值對應到新的灰階值,藉此拉開亮度分布
    """
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()

    # 忽略累積值為 0 的部分,避免除以 0
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    lookup_table = np.ma.filled(cdf_masked, 0).astype(np.uint8)

    equalized = lookup_table[gray]
    return equalized


# ---------------------------------------------------------
# 3. 效能與結果比較(跟 Quiz 1 同一套邏輯)
# ---------------------------------------------------------
def benchmark(func, *args, n_runs: int = 100):
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, np.array(times)


def compare_results(gray_a: np.ndarray, gray_b: np.ndarray, name_a: str, name_b: str):
    diff = np.abs(gray_a.astype(int) - gray_b.astype(int))
    print(f"\n--- {name_a} vs {name_b} 結果差異 ---")
    print(f"完全相同:       {np.array_equal(gray_a, gray_b)}")
    print(f"平均絕對差(MAE): {diff.mean():.6f}")
    print(f"最大差異:        {diff.max()}")
    print(f"差異像素比例:    {(diff > 0).mean() * 100:.2f}%")
    return diff


def print_timing(name: str, times_ms: np.ndarray):
    print(f"{name:20s}  平均: {times_ms.mean():8.4f} ms   標準差: {times_ms.std():7.4f} ms   "
          f"最快: {times_ms.min():8.4f} ms   最慢: {times_ms.max():8.4f} ms")


def save_results_table(t_opencv, t_numpy, diff, out_dir="."):
    """把速度比較與差異比較的數據畫成表格圖片"""
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.6))

    speed_cols = ["方法", "平均(ms)", "標準差(ms)", "最快(ms)", "最慢(ms)"]
    speed_rows = [
        ["OpenCV equalizeHist", f"{t_opencv.mean():.2f}", f"{t_opencv.std():.2f}",
         f"{t_opencv.min():.2f}", f"{t_opencv.max():.2f}"],
        ["NumPy 自行實作", f"{t_numpy.mean():.2f}", f"{t_numpy.std():.2f}",
         f"{t_numpy.min():.2f}", f"{t_numpy.max():.2f}"],
    ]
    axes[0].axis("off")
    axes[0].set_title("執行速度比較", fontsize=13, fontweight="bold", loc="left")
    tbl1 = axes[0].table(cellText=speed_rows, colLabels=speed_cols, cellLoc="center", loc="center")
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(10)
    tbl1.scale(1, 1.7)

    diff_cols = ["比較對象", "完全相同", "平均絕對差(MAE)", "最大差異", "差異像素比例"]
    diff_rows = [
        ["NumPy vs OpenCV", str(diff.max() == 0), f"{diff.mean():.6f}",
         str(diff.max()), f"{(diff > 0).mean() * 100:.2f}%"],
    ]
    axes[1].axis("off")
    axes[1].set_title("均化結果差異比較", fontsize=13, fontweight="bold", loc="left")
    tbl2 = axes[1].table(cellText=diff_rows, colLabels=diff_cols, cellLoc="center", loc="center")
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(10)
    tbl2.scale(1, 1.7)

    plt.tight_layout()
    path = os.path.join(out_dir, "results_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"已儲存結果表格圖: {path}")
    plt.close(fig)
    return path


# ---------------------------------------------------------
# 主函式:main.ipynb 呼叫的進入點
# ---------------------------------------------------------
def run(image_path: str, n_runs: int = 100, out_dir: str = "result/quiz2"):
    """
    執行 Quiz 2 的完整流程:讀圖 -> 轉灰階 -> 基礎題均化 -> 進階題效能與差異比較 -> 輸出圖表。

    Returns
    -------
    dict: 包含均化結果、統計數據與輸出檔案路徑,方便在 notebook 裡進一步檢查
    """
    os.makedirs(out_dir, exist_ok=True)

    img_rgb = load_image(image_path)
    gray = to_grayscale(img_rgb)

    # --- 基礎題:各方法各跑一次,拿到結果 ---
    eq_opencv = equalize_opencv(gray)
    eq_numpy = equalize_numpy(gray)

    # --- 進階題:重複執行 N 次,比較速度 ---
    print(f"\n================ 執行速度比較 (重複 {n_runs} 次取平均) ================")
    _, t_opencv = benchmark(equalize_opencv, gray, n_runs=n_runs)
    _, t_numpy = benchmark(equalize_numpy, gray, n_runs=n_runs)

    print_timing("OpenCV equalizeHist", t_opencv)
    print_timing("NumPy 自行實作", t_numpy)

    ratio = t_numpy.mean() / t_opencv.mean()
    if ratio >= 1:
        print(f"\nOpenCV 比 NumPy 快 {ratio:.2f} 倍")
    else:
        print(f"\nNumPy 比 OpenCV 快 {1 / ratio:.2f} 倍")

    # --- 進階題:比較均化結果的差異 ---
    print(f"\n================ 均化結果差異比較 ================")
    diff = compare_results(eq_numpy, eq_opencv, "NumPy 自行實作", "OpenCV")

    table_path = save_results_table(t_opencv, t_numpy, diff, out_dir=out_dir)

    # ---------------------------------------------------------
    # 視覺化 1:原圖、灰階圖、均化結果、差異熱圖
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("原始彩色影像")

    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("原始灰階影像")

    im0 = axes[0, 2].imshow(diff, cmap="hot", vmin=0, vmax=max(diff.max(), 1))
    axes[0, 2].set_title(f"NumPy vs OpenCV 差異\n(最大差異={diff.max()})")
    fig.colorbar(im0, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(eq_opencv, cmap="gray")
    axes[1, 0].set_title("OpenCV equalizeHist 結果")

    axes[1, 1].imshow(eq_numpy, cmap="gray")
    axes[1, 1].set_title("NumPy 自行實作結果")

    axes[1, 2].axis("off")

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]:
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n已儲存比較圖: {fig_path}")
    plt.show()

    # ---------------------------------------------------------
    # 視覺化 2:處理前後的灰階 Histogram
    # ---------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))

    axes2[0].hist(gray.flatten(), bins=256, range=(0, 256), color="#4C72B0")
    axes2[0].set_title("處理前:原始灰階 Histogram")
    axes2[0].set_xlabel("灰階值")
    axes2[0].set_ylabel("像素數量")

    axes2[1].hist(eq_opencv.flatten(), bins=256, range=(0, 256), color="#DD8452", alpha=0.6, label="OpenCV")
    axes2[1].hist(eq_numpy.flatten(), bins=256, range=(0, 256), color="#55A868", alpha=0.6, label="NumPy")
    axes2[1].set_title("處理後:均化後 Histogram")
    axes2[1].set_xlabel("灰階值")
    axes2[1].set_ylabel("像素數量")
    axes2[1].legend()

    plt.tight_layout()
    hist_path = os.path.join(out_dir, "histogram_comparison.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"已儲存直方圖比較圖: {hist_path}")
    plt.show()

    # 也把結果存成圖片檔
    gray_path = os.path.join(out_dir, "gray_original.png")
    opencv_path = os.path.join(out_dir, "equalized_opencv.png")
    numpy_path = os.path.join(out_dir, "equalized_numpy.png")
    cv2.imwrite(gray_path, gray)
    cv2.imwrite(opencv_path, eq_opencv)
    cv2.imwrite(numpy_path, eq_numpy)
    print(f"已儲存影像檔: {gray_path} / {opencv_path} / {numpy_path}")

    plt.close("all")

    return {
        "gray": gray,
        "eq_opencv": eq_opencv,
        "eq_numpy": eq_numpy,
        "timings": {"opencv": t_opencv, "numpy": t_numpy},
        "diff": diff,
        "output_files": [fig_path, hist_path, table_path, gray_path, opencv_path, numpy_path],
    }