"""
進階：VLM 影像擾動魯棒性測試（POPE 是非題 × 多種擾動）。

擾動種類（強度可調）
  clean        原圖
  dark/bright  亮度 ×0.35 / ×1.8         （模擬曝光不足/過曝）
  blur         Gaussian blur, radius=4    （模擬失焦/動態模糊）
  noise        Gaussian noise, σ=30       （模擬高 ISO 雜訊）
  jpeg         JPEG quality=8             （模擬壓縮失真）
  lowres       縮小至 1/4 再放大          （模擬低解析度）

指標（每種擾動各算一次）
  accuracy / precision / recall / F1，以及 yes_ratio
  —— yes_ratio 明顯高於 0.5 代表模型傾向「附和」，是幻覺的典型徵兆；
     觀察擾動後 yes_ratio 是否上升，可看出「看不清楚時模型會不會開始腦補」。

用法：python src/hw3/robustness_eval.py --model Qwen/Qwen2-VL-7B-Instruct --limit 300 [--adapter 路徑]
"""
import argparse
import io
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "hw2"))
from common.paths import DATA  # noqa: E402
from run_vlm import load_vlm, vlm_generate  # noqa: E402

COCO = os.path.join(DATA, "hw3", "coco")


def perturb(img, kind, rng):
    if kind == "clean":
        return img
    if kind == "dark":
        return ImageEnhance.Brightness(img).enhance(0.35)
    if kind == "bright":
        return ImageEnhance.Brightness(img).enhance(1.8)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=4))
    if kind == "noise":
        arr = np.asarray(img).astype(np.float32) + rng.normal(0, 30, (img.height, img.width, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=8)
        return Image.open(buf).convert("RGB")
    if kind == "lowres":
        return img.resize((img.width // 4, img.height // 4), Image.BILINEAR).resize(img.size, Image.BILINEAR)
    raise ValueError(kind)


PERTURBATIONS = ["clean", "dark", "bright", "blur", "noise", "jpeg", "lowres"]


def fetch(url, cache_dir):
    path = os.path.join(cache_dir, os.path.basename(url))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return Image.open(path).convert("RGB")


def pope_metrics(df):
    pred_yes = df["pred"] == "yes"
    gold_yes = df["label"] == "yes"
    tp = (pred_yes & gold_yes).sum()
    prec = tp / pred_yes.sum() if pred_yes.sum() else 0.0
    rec = tp / gold_yes.sum() if gold_yes.sum() else 0.0
    return pd.Series({"accuracy": (pred_yes == gold_yes).mean(), "precision": prec, "recall": rec,
                      "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "yes_ratio": pred_yes.mean()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--adapter", default=None, help="（選用）VLM-DPO 後的 LoRA adapter，比較微調前後魯棒性")
    ap.add_argument("--pope", default=os.path.join(COCO, "pope_eval.jsonl"))
    ap.add_argument("--perturbations", nargs="+", default=PERTURBATIONS, choices=PERTURBATIONS)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(DATA, "hw3", "results", "robustness"))
    args = ap.parse_args()

    cache = os.path.join(COCO, "images")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.pope) as f:
        qs = [json.loads(line) for line in f][: args.limit]

    model, processor = load_vlm(args.model, args.load_4bit)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()

    rows = []
    for kind in args.perturbations:
        rng = np.random.default_rng(0)          # 每種擾動固定亂數種子 → 可重現
        for k, q in enumerate(qs):
            img = perturb(fetch(q["image_url"], cache), kind, rng)
            resp, _ = vlm_generate(model, processor, img, q["question"], max_new_tokens=8)
            pred = "yes" if resp.strip().lower().startswith("yes") else "no"
            rows.append({**q, "perturbation": kind, "response": resp, "pred": pred})
        print(f"  {kind}: 完成 {len(qs)} 題")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "robustness_raw.csv"), index=False)
    table = df.groupby(["perturbation", "mode"]).apply(pope_metrics).round(3)
    overall = df.groupby("perturbation").apply(pope_metrics).round(3).reindex(args.perturbations)
    overall["acc_drop_vs_clean"] = (overall["accuracy"] - overall.loc["clean", "accuracy"]).round(3) if "clean" in overall.index else None
    table.to_csv(os.path.join(args.out_dir, "robustness_by_mode.csv"))
    overall.to_csv(os.path.join(args.out_dir, "robustness_overall.csv"))
    print("\n" + overall.to_string())


if __name__ == "__main__":
    main()
