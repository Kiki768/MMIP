"""
進階：比較微調前後「描述圖片」時提及不存在物件的比例（CHAIR 風格，以 COCO 標註判定）。

  只看 COCO 80 類物件：描述中提到某類別、但該圖完全沒有這類標註 → 算一次幻覺。
  halluc_object_rate = 幻覺物件數 ÷ 提及的 COCO 物件總數（越低越好）
  halluc_image_rate  = 至少有一個幻覺物件的圖片比例（越低越好）
  mentioned_per_image = 平均提及物件數（防止模型靠「什麼都不說」來壓低幻覺）

只用訓練時排除的圖片。base 與微調後共用同一份權重，以 disable_adapter 切換。

用法：python src/hw3/eval_vlm_hallucination.py --adapter data/hw3/adapters/qwen2vl-dpo-halluc
"""
import argparse
import collections
import json
import os
import re
import sys

import pandas as pd
import torch
from peft import PeftModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "hw2"))
from common.paths import DATA  # noqa: E402
from robustness_eval import fetch  # noqa: E402
from run_vlm import load_vlm, vlm_generate  # noqa: E402

COCO = os.path.join(DATA, "hw3", "coco")
PROMPT = "Describe this image in detail, listing the objects you can see."


def mentioned(text, categories):
    t = text.lower()
    return {c for c in categories if re.search(rf"\b{re.escape(c)}(s|es)?\b", t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--adapter", default=os.path.join(DATA, "hw3", "adapters", "qwen2vl-dpo-halluc"))
    ap.add_argument("--holdout-questions", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out-dir", default=os.path.join(DATA, "hw3", "results", "vlm_dpo"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(COCO, "pope_eval.jsonl"), encoding="utf-8") as f:
        pope = [json.loads(line) for line in f][: args.holdout_questions]
    images = list(dict.fromkeys((q["image_id"], q["image_url"]) for q in pope))

    with open(os.path.join(COCO, "annotations", "instances_val2017.json")) as f:
        inst = json.load(f)
    cat = {c["id"]: c["name"] for c in inst["categories"]}
    present = collections.defaultdict(set)
    for a in inst["annotations"]:
        present[a["image_id"]].add(cat[a["category_id"]])
    categories = sorted(cat.values())

    base, processor = load_vlm(args.model)
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    cache = os.path.join(COCO, "images")

    rows = []
    for tag in ("base", "dpo"):
        for iid, url in images:
            img = fetch(url, cache)
            if tag == "base":
                with model.disable_adapter():
                    resp, _ = vlm_generate(model, processor, img, PROMPT, args.max_new_tokens)
            else:
                resp, _ = vlm_generate(model, processor, img, PROMPT, args.max_new_tokens)
            found = mentioned(resp, categories)
            bad = found - present[iid]
            rows.append({"model": tag, "image_id": iid, "response": resp, "n_mentioned": len(found),
                         "n_halluc": len(bad), "halluc_objects": ", ".join(sorted(bad))})
        print(f"  {tag}: 完成 {len(images)} 張", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "caption_halluc_raw.csv"), index=False, encoding="utf-8-sig")
    g = df.groupby("model", sort=False)
    summary = pd.DataFrame({"images": g.size(),
                            "halluc_object_rate": g["n_halluc"].sum() / g["n_mentioned"].sum(),
                            "halluc_image_rate": g["n_halluc"].apply(lambda s: (s > 0).mean()),
                            "mentioned_per_image": g["n_mentioned"].mean()}).round(3)
    summary.to_csv(os.path.join(args.out_dir, "caption_halluc_summary.csv"), encoding="utf-8-sig")
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
