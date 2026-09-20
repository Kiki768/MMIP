import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 讓中文字在圖表上正常顯示(如果你的電腦沒有微軟正黑體,
# 可以換成 "PingFang TC"(Mac)或 "Noto Sans CJK TC"(Linux))
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------
# 0. 讀取影像
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
    # print(f"已讀取圖片: {image_path}, 尺寸 = {img_rgb.shape}")
    return img_rgb


# ---------------------------------------------------------
# 1. 基礎題:RGB -> 灰階 (直接用 OpenCV)
# ---------------------------------------------------------
def rgb_to_gray_opencv(img_rgb: np.ndarray) -> np.ndarray:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray


# ---------------------------------------------------------
# 2. 進階題:自行用 NumPy 實作灰階轉換(兩種常見公式)
# ---------------------------------------------------------
def rgb_to_gray_numpy_average(img_rgb: np.ndarray) -> np.ndarray:
    """簡單平均法: Gray = (R + G + B) / 3"""
    img_float = img_rgb.astype(np.float64)
    gray = img_float.mean(axis=2)
    return np.round(gray).astype(np.uint8)


def rgb_to_gray_numpy_weighted(img_rgb: np.ndarray) -> np.ndarray:
    """
    ITU-R BT.601 加權法(OpenCV 內部實際使用的公式):
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    img_float = img_rgb.astype(np.float64)
    weights = np.array([0.299, 0.587, 0.114])
    gray = np.dot(img_float, weights)
    return np.round(gray).astype(np.uint8)


# ---------------------------------------------------------
# 3. 效能與結果比較
# ---------------------------------------------------------
def benchmark(func, *args, n_runs: int = 100):
    """執行 n_runs 次,回傳 (最後一次的結果, 每次耗時的 list(單位: ms))"""
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


def save_results_table(t_opencv, t_avg, t_weighted, diff_weighted_cv, diff_avg_cv, out_dir="."):
    """把速度比較與差異比較的數據畫成表格圖片,方便直接放進報告"""
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.3))

    speed_cols = ["方法", "平均(ms)", "標準差(ms)", "最快(ms)", "最慢(ms)"]
    speed_rows = [
        ["OpenCV", f"{t_opencv.mean():.2f}", f"{t_opencv.std():.2f}",
         f"{t_opencv.min():.2f}", f"{t_opencv.max():.2f}"],
        ["NumPy (平均法)", f"{t_avg.mean():.2f}", f"{t_avg.std():.2f}",
         f"{t_avg.min():.2f}", f"{t_avg.max():.2f}"],
        ["NumPy (加權法)", f"{t_weighted.mean():.2f}", f"{t_weighted.std():.2f}",
         f"{t_weighted.min():.2f}", f"{t_weighted.max():.2f}"],
    ]
    axes[0].axis("off")
    axes[0].set_title("執行速度比較", fontsize=13, fontweight="bold", loc="left")
    tbl1 = axes[0].table(cellText=speed_rows, colLabels=speed_cols, cellLoc="center", loc="center")
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(10)
    tbl1.scale(1, 1.7)

    diff_cols = ["比較對象", "完全相同", "平均絕對差(MAE)", "最大差異", "差異像素比例"]
    diff_rows = [
        ["加權法 vs OpenCV", str(diff_weighted_cv.max() == 0),
         f"{diff_weighted_cv.mean():.6f}", str(diff_weighted_cv.max()),
         f"{(diff_weighted_cv > 0).mean() * 100:.2f}%"],
        ["平均法 vs OpenCV", str(diff_avg_cv.max() == 0),
         f"{diff_avg_cv.mean():.6f}", str(diff_avg_cv.max()),
         f"{(diff_avg_cv > 0).mean() * 100:.2f}%"],
    ]
    axes[1].axis("off")
    axes[1].set_title("轉換結果差異比較", fontsize=13, fontweight="bold", loc="left")
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
def run(image_path: str, n_runs: int = 100, out_dir: str = "result/quiz1"):
    """
    執行 Quiz 1 的完整流程:讀圖 -> 基礎題轉換 -> 進階題效能與差異比較 -> 輸出圖表。

    Returns
    -------
    dict: 包含灰階結果、統計數據與輸出檔案路徑,方便在 notebook 裡進一步檢查
    """
    os.makedirs(out_dir, exist_ok=True)

    img_rgb = load_image(image_path)

    gray_opencv = rgb_to_gray_opencv(img_rgb)
    gray_avg = rgb_to_gray_numpy_average(img_rgb)
    gray_weighted = rgb_to_gray_numpy_weighted(img_rgb)

    print(f"\n================ 執行速度比較 (重複 {n_runs} 次取平均) ================")
    _, t_opencv = benchmark(rgb_to_gray_opencv, img_rgb, n_runs=n_runs)
    _, t_avg = benchmark(rgb_to_gray_numpy_average, img_rgb, n_runs=n_runs)
    _, t_weighted = benchmark(rgb_to_gray_numpy_weighted, img_rgb, n_runs=n_runs)

    print_timing("OpenCV", t_opencv)
    print_timing("NumPy (平均法)", t_avg)
    print_timing("NumPy (加權法)", t_weighted)

    speedup = t_avg.mean() / t_opencv.mean()
    print(f"\nOpenCV 比 NumPy(平均法) 快 {speedup:.2f} 倍")
    speedup2 = t_weighted.mean() / t_opencv.mean()
    print(f"OpenCV 比 NumPy(加權法) 快 {speedup2:.2f} 倍")

    print(f"\n================ 轉換結果差異比較 ================")
    diff_weighted_cv = compare_results(gray_weighted, gray_opencv, "NumPy(加權法)", "OpenCV")
    diff_avg_cv = compare_results(gray_avg, gray_opencv, "NumPy(平均法)", "OpenCV")

    table_path = save_results_table(t_opencv, t_avg, t_weighted, diff_weighted_cv, diff_avg_cv, out_dir=out_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("原始彩色影像")

    axes[0, 1].imshow(gray_opencv, cmap="gray")
    axes[0, 1].set_title("OpenCV cvtColor")

    axes[0, 2].imshow(gray_weighted, cmap="gray")
    axes[0, 2].set_title("NumPy 加權法 (BT.601)")

    axes[1, 0].imshow(gray_avg, cmap="gray")
    axes[1, 0].set_title("NumPy 簡單平均法")

    im1 = axes[1, 1].imshow(diff_weighted_cv, cmap="hot", vmin=0, vmax=max(diff_weighted_cv.max(), 1))
    axes[1, 1].set_title(f"加權法 vs OpenCV 差異\n(最大差異={diff_weighted_cv.max()})")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046)

    im2 = axes[1, 2].imshow(diff_avg_cv, cmap="hot", vmin=0, vmax=max(diff_avg_cv.max(), 1))
    axes[1, 2].set_title(f"平均法 vs OpenCV 差異\n(最大差異={diff_avg_cv.max()})")
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "comparison_result.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n已儲存比較圖: {fig_path}")
    plt.show()

    gray_opencv_path = os.path.join(out_dir, "gray_opencv.png")
    gray_weighted_path = os.path.join(out_dir, "gray_numpy_weighted.png")
    gray_avg_path = os.path.join(out_dir, "gray_numpy_average.png")
    cv2.imwrite(gray_opencv_path, gray_opencv)
    cv2.imwrite(gray_weighted_path, gray_weighted)
    cv2.imwrite(gray_avg_path, gray_avg)
    print(f"已儲存灰階影像檔: {gray_opencv_path} / {gray_weighted_path} / {gray_avg_path}")

    plt.close("all")

    return {
        "gray_opencv": gray_opencv,
        "gray_avg": gray_avg,
        "gray_weighted": gray_weighted,
        "timings": {"opencv": t_opencv, "average": t_avg, "weighted": t_weighted},
        "diffs": {"weighted_vs_opencv": diff_weighted_cv, "average_vs_opencv": diff_avg_cv},
        "output_files": [fig_path, table_path, gray_opencv_path, gray_weighted_path, gray_avg_path],
    }