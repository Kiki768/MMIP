import os
import cv2
import numpy as np


def order_points(pts):
    """將四個頂點依序排列為：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")

    # 左上角總和最小，右下角總和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上角差值最小 (y - x)，左下角差值最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """執行透視校正變換"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 計算新影像寬度
    width_a = np.hypot(br[0] - bl[0], br[1] - bl[1])
    width_b = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    max_width = max(int(width_a), int(width_b))

    # 計算新影像高度
    height_a = np.hypot(tr[0] - br[0], tr[1] - br[1])
    height_b = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    max_height = max(int(height_a), int(height_b))

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    # 計算透視變換矩陣並進行變換
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def auto_detect_contour(image):
    """
    增強版自動偵測：
    1. 雙邊濾波去除布料細緻網紋
    2. 結合形態學閉運算消除內部文字干擾
    3. 利用凸包與漸進式多邊形擬合解決「圓角」問題
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 雙邊濾波：平滑背景網格紋理與雜訊，同時保留主要邊界
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Otsu 自適應二值化 (白底目標 vs 深底背景)
    _, thresh = cv2.threshold(
        filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. 形態學閉運算：填補卡片內部文字產生的孔洞，融合成一整塊大輪廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. 尋找輪廓
    cnts, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        return None

    # 取面積最大的前三個輪廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    h, w = image.shape[:2]
    min_area = h * w * 0.05  # 面積需佔畫面 5% 以上

    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue

        # 取輪廓凸包 (Convex Hull) 撫平圓角引起的鋸齒
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)

        # 漸進放寬近似精度 epsilon (從 0.02 到 0.05)，嘗試收斂至 4 個頂點
        found_box = None
        for eps_ratio in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(hull, eps_ratio * peri, True)
            if len(approx) == 4:
                found_box = approx.reshape(4, 2)
                break

        if found_box is not None:
            return found_box.astype("float32")

        # 若圓角過大依然無法恰好擬合出 4 點，使用最小外接斜矩形作為穩健備援
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        return box.astype("float32")

    return None


def run(image_paths, out_dir="result/quiz3"):
    """批次執行梯形校正並輸出成果"""
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[警告] 無法讀取檔案: {img_path}")
            continue

        pts = auto_detect_contour(img)

        if pts is not None:
            warped = four_point_transform(img, pts)
            out_path = os.path.join(out_dir, f"corrected_{file_name}")
            cv2.imwrite(out_path, warped)
            results[file_name] = {"status": "success", "output": out_path}
            print(f"[成功] {file_name} 校正完成 -> {out_path}")
        else:
            out_path = os.path.join(out_dir, f"failed_{file_name}")
            cv2.imwrite(out_path, img)
            results[file_name] = {
                "status": "failed",
                "reason": "未偵測到有效邊界",
            }
            print(f"[失敗] {file_name} 無法自動辨識四邊形邊界")

    return results