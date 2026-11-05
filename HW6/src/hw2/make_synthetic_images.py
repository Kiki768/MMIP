"""
產生「標準答案 100% 已知」的合成測試圖（圖表 / OCR / 物件位置），並下載 COCO 自然影像。

為什麼要自己畫圖？
  - 圖表、表格、收據的數值完全由程式決定 → 可以做 Exact-Match 之類的「客觀量化指標」，
    不必全靠 LLM 評審。
  - 幾何圖形的座標已知 → 可以計算 IoU / 中心點誤差，評估定位能力。

用法：python src/hw2/make_synthetic_images.py
"""
import os
import sys
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import DATA, FONT as FONT_PATH  # noqa: E402

IMG_DIR = os.path.join(DATA, "hw2", "images")
FONT_URL = "https://github.com/notofonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"

COCO_IDS = [
    "000000039769", "000000000139", "000000000285", "000000000632", "000000000724",
    "000000000776", "000000001000", "000000001268", "000000002153", "000000002261",
    "000000003156", "000000005001",
]


def ensure_font():
    if not os.path.exists(FONT_PATH):
        os.makedirs(os.path.dirname(FONT_PATH), exist_ok=True)
        print("下載繁體中文字型 Noto Sans CJK TC ...")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()


def download_coco():
    for cid in COCO_IDS:
        path = os.path.join(IMG_DIR, f"coco_{cid}.jpg")
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{cid}.jpg", path)


def bar_chart():
    # 標準答案：最高 = 第三季 (Q3) 420；最低 = Q1 180；總和 = 1200
    quarters, sales = ["Q1", "Q2", "Q3", "Q4"], [180, 260, 420, 340]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    bars = ax.bar(quarters, sales, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.bar_label(bars)
    ax.set_title("2024 年各季營收（萬元）")
    ax.set_ylabel("營收（萬元）")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "chart_bar.png"))
    plt.close(fig)


def line_chart():
    # 標準答案：台北 7 月最高 30°C；高雄 1 月 19°C；兩線在 1~7 月皆為台北 < 高雄 (除 7 月相等)
    months = [1, 2, 3, 4, 5, 6, 7]
    taipei = [16, 17, 19, 22, 26, 28, 30]
    kaohsiung = [19, 20, 23, 25, 27, 29, 30]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(months, taipei, marker="o", label="台北")
    ax.plot(months, kaohsiung, marker="s", label="高雄")
    ax.set_xticks(months)
    ax.set_xlabel("月份")
    ax.set_ylabel("平均氣溫 (°C)")
    ax.set_title("1–7 月平均氣溫")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "chart_line.png"))
    plt.close(fig)


def pie_chart():
    # 標準答案：交通 35%、餐飲 25%、住宿 30%、其他 10%
    labels, values = ["交通", "餐飲", "住宿", "其他"], [35, 25, 30, 10]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.pie(values, labels=labels, autopct="%d%%", startangle=90)
    ax.set_title("旅遊預算分配")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "chart_pie.png"))
    plt.close(fig)


def table_image():
    rows = [["姓名", "國文", "英文", "數學"],
            ["王小明", "85", "92", "78"],
            ["李美華", "90", "88", "95"],
            ["陳志強", "72", "80", "88"]]
    fig, ax = plt.subplots(figsize=(5, 2.2), dpi=130)
    ax.axis("off")
    t = ax.table(cellText=rows, loc="center", cellLoc="center")
    t.scale(1, 1.8)
    t.set_fontsize(12)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "table_scores.png"))
    plt.close(fig)


def receipt_image():
    font = ImageFont.truetype(FONT_PATH, 26)
    big = ImageFont.truetype(FONT_PATH, 34)
    img = Image.new("RGB", (520, 560), "white")
    d = ImageDraw.Draw(img)
    lines = [
        ("好日子咖啡館", big),
        ("2024/03/15  14:32", font),
        ("------------------------------", font),
        ("拿鐵 x2          NT$ 240", font),
        ("司康 x1          NT$  85", font),
        ("檸檬塔 x1        NT$ 110", font),
        ("------------------------------", font),
        ("合計             NT$ 435", font),
        ("付款方式：信用卡", font),
        ("發票號碼：AB-12345678", font),
    ]
    y = 30
    for text, f in lines:
        d.text((30, y), text, fill="black", font=f)
        y += 50
    img.save(os.path.join(IMG_DIR, "ocr_receipt.png"))


def sign_image():
    img = Image.new("RGB", (800, 450), (20, 90, 60))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 780, 430], outline="white", width=6)
    d.text((60, 60), "國立示範大學 圖書館", fill="white", font=ImageFont.truetype(FONT_PATH, 52))
    f = ImageFont.truetype(FONT_PATH, 34)
    d.text((60, 170), "開放時間：週一至週五 08:00–22:00", fill="white", font=f)
    d.text((60, 230), "週六 09:00–17:00　週日休館", fill="white", font=f)
    d.text((60, 310), "請勿飲食 · 保持安靜", fill=(255, 220, 80), font=f)
    img.save(os.path.join(IMG_DIR, "ocr_sign.png"))


def shapes_image():
    # 標準答案（像素座標, 畫布 600x400）：
    #   紅色圓形 bbox=[40,40,160,160]（左上）
    #   藍色正方形 bbox=[420,60,540,180]（右上）
    #   綠色三角形 bbox≈[240,240,360,360]（下方中央）
    img = Image.new("RGB", (600, 400), "white")
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, 160, 160], fill=(220, 40, 40))
    d.rectangle([420, 60, 540, 180], fill=(40, 80, 220))
    d.polygon([(300, 240), (240, 360), (360, 360)], fill=(40, 170, 70))
    img.save(os.path.join(IMG_DIR, "shapes_layout.png"))


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    ensure_font()
    download_coco()
    for fn in (bar_chart, line_chart, pie_chart, table_image, receipt_image, sign_image, shapes_image):
        fn()
    print("完成，圖片位於", IMG_DIR)
