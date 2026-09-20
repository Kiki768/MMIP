# -*- coding: utf-8 -*-
"""
Quiz 4: 影像拼接 (Image Stitching)
基礎: 選兩張有重疊區域的影像,用 SIFT 等特徵偵測與匹配方法完成影像拼接
進階:
  1. 調整影像的亮度、拍攝角度或重疊範圍,觀察不同條件下的拼接效果,
     找出演算法開始無法成功拼接的條件,並說明可能原因
     (提供 test_conditions() 自動從一組基準影像產生多種條件的變化版本,
      不需要真的拍很多組照片,就能有系統地測試)
  2. 嘗試在拼接前加入前處理方法(CLAHE 對比強化),提升特徵偵測與匹配的穩定性

在 main.ipynb 裡用:
    from src import quiz4

    # 基礎題:單組影像拼接
    quiz4.run([("dataset/left.jpg", "dataset/right.jpg")], out_dir="result/quiz4")

    # 進階題:自動測試不同條件(亮度、角度、重疊範圍)下的拼接效果
    quiz4.test_conditions("dataset/left.jpg", "dataset/right.jpg", out_dir="result/quiz4")
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------
# 1. 基礎題:SIFT 特徵偵測與匹配
# ---------------------------------------------------------
def detect_sift_features(img_rgb: np.ndarray):
    """偵測 SIFT 關鍵點與描述子(descriptor)"""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75):
    """
    用 KNN(k=2)比對兩組描述子,再套用 Lowe's ratio test 篩選出可靠的配對。
    ratio 越小,篩選越嚴格(留下的配對越少但越可靠)。
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches


def estimate_homography(kp1, kp2, good_matches, min_matches: int = 10):
    """
    用配對點估計 Homography 矩陣(RANSAC 排除離群點)。

    Returns
    -------
    H : np.ndarray or None
    inlier_ratio : float  (RANSAC 認定為合理配對的比例,越高代表匹配越可靠)
    n_matches : int  (送進 RANSAC 的配對點數量)
    """
    n_matches = len(good_matches)
    if n_matches < min_matches:
        return None, 0.0, n_matches

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None, 0.0, n_matches

    inlier_ratio = float(mask.sum()) / n_matches
    return H, inlier_ratio, n_matches


# ---------------------------------------------------------
# 2. 前處理:CLAHE 對比強化(進階題第 2 點)
# ---------------------------------------------------------
def preprocess_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """
    用 CLAHE(局部自適應直方圖均衡化,概念上是 Quiz 2 直方圖均衡化的進階版)
    強化對比,讓低對比、偏暗的影像也能偵測到足夠的特徵點。
    只對亮度(Y 通道)做強化,保留原始色彩,做法跟 Quiz 2 討論過的彩色版
    Histogram Equalization 原理一樣。
    """
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    merged = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)


# ---------------------------------------------------------
# 3. 影像拼接:把兩張圖縫合成一張全景圖
# ---------------------------------------------------------
def warp_and_stitch(img1_rgb: np.ndarray, img2_rgb: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    用 Homography H 把 img1 扭曲到 img2 的座標系,再跟 img2 合併成一張全景圖。
    自動計算輸出畫布大小,確保扭曲後的 img1 不會被裁掉。
    """
    h1, w1 = img1_rgb.shape[:2]
    h2, w2 = img2_rgb.shape[:2]

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    warped_corners1 = cv2.perspectiveTransform(corners1, H)

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([warped_corners1, corners2], axis=0)

    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    output_size = (x_max - x_min, y_max - y_min)

    warped_img1 = cv2.warpPerspective(img1_rgb, translation @ H, output_size)

    canvas = warped_img1.copy()
    x_off, y_off = -x_min, -y_min
    region = canvas[y_off:y_off + h2, x_off:x_off + w2]

    # 兩張圖都有內容的地方,用 img2 蓋過去(避免接縫處被扭曲的 img1 弄髒)
    mask2 = np.any(img2_rgb > 0, axis=2)
    region[mask2] = img2_rgb[mask2]
    canvas[y_off:y_off + h2, x_off:x_off + w2] = region

    return canvas


# ---------------------------------------------------------
# 4. 單組影像的完整流程 + 視覺化
# ---------------------------------------------------------
def stitch_pair(image_path1: str, image_path2: str, out_dir: str,
                 use_clahe: bool = False, label: str = None) -> dict:
    """對一組影像執行完整拼接流程,輸出視覺化圖表,回傳診斷結果 dict"""
    name1 = os.path.splitext(os.path.basename(image_path1))[0]
    name2 = os.path.splitext(os.path.basename(image_path2))[0]
    label = label or f"{name1}_{name2}"

    img1 = load_image(image_path1)
    img2 = load_image(image_path2)

    proc1 = preprocess_clahe(img1) if use_clahe else img1
    proc2 = preprocess_clahe(img2) if use_clahe else img2

    kp1, desc1 = detect_sift_features(proc1)
    kp2, desc2 = detect_sift_features(proc2)

    good_matches = match_features(desc1, desc2)
    H, inlier_ratio, n_matches = estimate_homography(kp1, kp2, good_matches)

    result = {
        "label": label, "success": False,
        "n_keypoints1": len(kp1), "n_keypoints2": len(kp2),
        "n_good_matches": n_matches, "n_inliers": 0, "inlier_ratio": 0.0,
        "panorama": None, "output_file": None,
    }

    if H is None:
        print(f"[{label}] 拼接失敗:特徵點({len(kp1)}, {len(kp2)})或可靠配對數"
              f"({n_matches})不足,無法估計 Homography")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img1)
        axes[0].set_title(f"影像 1 (偵測到 {len(kp1)} 個特徵點)")
        axes[1].imshow(img2)
        axes[1].set_title(f"影像 2 (偵測到 {len(kp2)} 個特徵點)\n拼接失敗:可靠配對數={n_matches}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fail_path = os.path.join(out_dir, f"{label}_failed.png")
        plt.savefig(fail_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        result["output_file"] = fail_path
        return result

    n_inliers = round(inlier_ratio * n_matches)
    panorama = warp_and_stitch(img1, img2, H)

    # 畫配對關係圖(只挑前 60 組最佳配對,避免畫面太亂看不清楚)
    match_vis = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR), kp1,
        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR), kp2,
        sorted(good_matches, key=lambda m: m.distance)[:60], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    match_vis = cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))
    axes[0].imshow(match_vis)
    axes[0].set_title(
        f"{label}:特徵匹配\n"
        f"(可靠配對數={n_matches}, RANSAC 內點比例={inlier_ratio:.2f}, 內點數={n_inliers})"
    )
    axes[1].imshow(panorama)
    axes[1].set_title(f"{label}:拼接結果")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{label}_result.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(
        f"[{label}] 拼接成功   特徵點=({len(kp1)}, {len(kp2)})   "
        f"可靠配對數={n_matches}   RANSAC內點比例={inlier_ratio:.2f}"
    )

    result.update({
        "success": True, "n_inliers": n_inliers, "inlier_ratio": inlier_ratio,
        "panorama": panorama, "output_file": out_path,
    })
    return result


def save_summary_table(results: list, out_dir: str, filename: str = "summary_table.png") -> str:
    """把多組拼接結果整理成一張總表圖片"""
    cols = ["條件", "拼接成功", "特徵點數(img1,img2)", "可靠配對數", "RANSAC內點比例"]
    rows = []
    for r in results:
        rows.append([
            r["label"],
            "成功" if r["success"] else "失敗",
            f"({r['n_keypoints1']}, {r['n_keypoints2']})",
            str(r["n_good_matches"]),
            f"{r['inlier_ratio']:.2f}" if r["success"] else "-",
        ])

    fig, ax = plt.subplots(figsize=(11, 0.8 + 0.5 * len(rows)))
    ax.axis("off")
    # ax.set_title("不同條件下拼接結果總表", fontsize=13, fontweight="bold", loc="left")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.7)
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"已儲存總表: {path}")
    return path


# ---------------------------------------------------------
# 5. 主函式:main.ipynb 呼叫的進入點(基礎題)
# ---------------------------------------------------------
def run(image_pairs, out_dir: str = "result/quiz4", use_clahe: bool = False) -> dict:
    """
    Quiz 4 主流程:對一組或多組影像配對執行 SIFT 特徵拼接。

    Parameters
    ----------
    image_pairs : tuple(str, str) 或 list[tuple(str, str)]
        一組或多組 (影像1路徑, 影像2路徑)
    out_dir : str
        輸出圖表資料夾,預設 "result/quiz4"
    use_clahe : bool
        是否在拼接前先用 CLAHE 強化對比(進階題第 2 點)

    Returns
    -------
    dict: 每組的處理結果、輸出檔案路徑列表
    """
    if isinstance(image_pairs, tuple):
        image_pairs = [image_pairs]

    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i, (path1, path2) in enumerate(image_pairs):
        label = "basic" if len(image_pairs) == 1 else f"basic_{i + 1}"
        result = stitch_pair(path1, path2, out_dir, use_clahe=use_clahe, label=label)
        results.append(result)

    n_success = sum(r["success"] for r in results)
    # print(f"\n================ 總結 ================")
    # print(f"總共處理 {len(results)} 組影像,成功拼接 {n_success} 組"
    #       f"({n_success / len(results) * 100:.1f}%)")

    table_path = save_summary_table(results, out_dir, filename="summary_table_basic.png")

    return {
        "results": results,
        "success_rate": n_success / len(results),
        "output_files": [r["output_file"] for r in results if r["output_file"]] + [table_path],
    }


# ---------------------------------------------------------
# 6. 進階題第 1 點:自動測試不同條件下的拼接效果
# ---------------------------------------------------------
def _adjust_brightness(img_rgb: np.ndarray, factor: float) -> np.ndarray:
    """factor > 1 變亮,factor < 1 變暗"""
    adjusted = img_rgb.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _rotate_image(img_rgb: np.ndarray, angle_deg: float) -> np.ndarray:
    """模擬拍攝角度差異:以圖片中心旋轉指定角度"""
    h, w = img_rgb.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img_rgb, M, (w, h), borderValue=(0, 0, 0))


def _reduce_overlap(img_rgb: np.ndarray, keep_ratio: float) -> np.ndarray:
    """
    模擬重疊範圍變小:把 img2 左側(跟 img1 重疊的那一側)裁掉一部分,
    keep_ratio 是保留的寬度比例,越小代表重疊區域越少。
    """
    h, w = img_rgb.shape[:2]
    start = int(w * (1 - keep_ratio))
    return img_rgb[:, start:]


def test_conditions(image_path1: str, image_path2: str, out_dir: str = "result/quiz4") -> dict:
    """
    進階題第 1 點:用一組基準影像,自動產生多種條件(亮度、角度、重疊範圍)的
    變化版本,測試演算法在哪個條件開始無法成功拼接。

    不需要真的拍很多組照片——用一組品質好的基準照片,程式自動生成受控的
    測試變化版本,結果更容易比較(只變動一個條件,其他都固定)。
    """
    os.makedirs(out_dir, exist_ok=True)
    img1 = load_image(image_path1)
    img2 = load_image(image_path2)

    conditions = [
        ("亮度-變暗50%", img1, _adjust_brightness(img2, 0.5)),
        ("亮度-變暗80%", img1, _adjust_brightness(img2, 0.2)),
        ("亮度-變暗90%", img1, _adjust_brightness(img2, 0.1)),
        ("亮度-變亮150%", img1, _adjust_brightness(img2, 1.5)),
        ("亮度-變亮200%", img1, _adjust_brightness(img2, 2.0)),
        ("角度-旋轉20度", img1, _rotate_image(img2, 20)),
        ("角度-旋轉40度", img1, _rotate_image(img2, 40)),
        ("角度-旋轉60度", img1, _rotate_image(img2, 60)),
        ("角度-旋轉80度", img1, _rotate_image(img2, 80)),
        ("角度-旋轉100度", img1, _rotate_image(img2, 100)),
        ("重疊-保留80%重疊區", img1, _reduce_overlap(img2, 0.8)),
        ("重疊-保留60%重疊區", img1, _reduce_overlap(img2, 0.6)),
        ("重疊-保留40%重疊區", img1, _reduce_overlap(img2, 0.4)),
    ]

    results = []
    for label, im1, im2 in conditions:
        tmp1 = os.path.join(out_dir, "_tmp1.jpg")
        tmp2 = os.path.join(out_dir, "_tmp2.jpg")
        cv2.imwrite(tmp1, cv2.cvtColor(im1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(tmp2, cv2.cvtColor(im2, cv2.COLOR_RGB2BGR))
        result = stitch_pair(tmp1, tmp2, out_dir, label=label)
        results.append(result)
        os.remove(tmp1)
        os.remove(tmp2)

    n_success = sum(r["success"] for r in results)
    print(f"\n================ 條件測試總結 ================")
    print(f"總共測試 {len(results)} 種條件,成功 {n_success} 種"
          f"({n_success / len(results) * 100:.1f}%)")

    table_path = save_summary_table(results, out_dir, filename="summary_table_conditions.png")

    return {
        "results": results,
        "success_rate": n_success / len(results),
        "output_files": [r["output_file"] for r in results if r["output_file"]] + [table_path],
    }