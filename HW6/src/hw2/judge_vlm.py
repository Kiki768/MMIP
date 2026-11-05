"""
作業二：VLM 輸出的「客觀量化指標」＋「LLM-as-a-Judge」評分。

一、客觀指標（不需 API，可重現）
  - keyword_recall：每題的關鍵詞組（同義詞任一出現即算命中）命中比例
  - bbox_mIoU     ：LOC-04 合成圖，依顏色配對預測框與真值框計算平均 IoU
  - halluc_trap   ：幻覺陷阱題是否正確否認不存在的物件

二、單模型絕對評分（Pointwise，評審可看到圖片 + 參考答案）
  accuracy(0.4) / completeness(0.2) / hallucination(0.25) / instruction_following(0.15)，各 1~10 分

三、成對比較（Pairwise，A/B 兩種順序各評一次以消除位置偏誤）
  兩次結論一致才算勝，否則記為平手

用法：python src/hw2/judge_vlm.py --out-dir data/hw2/results [--no-pairwise] [--no-image]
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.judge import MODEL as JUDGE_MODEL, judge  # noqa: E402
from common.paths import DATA  # noqa: E402

HW_DATA = os.path.join(DATA, "hw2")

# ----------------------------------------------------------------------------- Rubric
POINTWISE_SYSTEM = """你是一位嚴格的視覺語言模型評審。你會看到【圖片】、【使用者指令】、【參考答案】與【模型回答】。
參考答案由出題者人工確認，代表圖片的真實內容；若模型描述了參考答案沒提到但圖片中確實存在的內容，不算錯。
請依下列維度各給 1~10 的整數：

- accuracy（準確性）：回答中的物件、類別、數值、文字、位置是否與圖片/參考答案一致。
    10=完全正確；7=主要正確、少數細節錯；4=關鍵資訊錯誤；1=完全錯誤。
- completeness（完整度）：是否回答了指令中「所有」子問題，描述題是否涵蓋參考答案的主要細節。
- hallucination（無幻覺程度）：10=沒有任何捏造；每捏造一個圖中不存在的物件/文字/數字扣 2~3 分。
    幻覺陷阱題（詢問不存在的物件）若回答「存在」，此項最多 2 分。
- instruction_following（指令遵循）：是否依要求的格式（JSON、Markdown 表格、選項字母）與語言（繁體中文）作答。

只輸出 JSON：
{"accuracy": int, "completeness": int, "hallucination": int, "instruction_following": int,
 "hallucinated_items": ["捏造的物件或內容，沒有則為空陣列"], "rationale": "60 字以內"}"""

PAIRWISE_SYSTEM = """你是一位公正的視覺語言模型評審。請根據【圖片】、【使用者指令】與【參考答案】，
比較【回答 A】與【回答 B】哪一個整體品質較好（優先順序：正確性 > 無幻覺 > 完整度 > 格式）。
不要受回答長度或出現順序影響。只輸出 JSON：{"winner": "A" | "B" | "tie", "reason": "40 字以內"}"""

WEIGHTS = {"accuracy": 0.4, "completeness": 0.2, "hallucination": 0.25, "instruction_following": 0.15}


# ----------------------------------------------------------------------------- 客觀指標
def keyword_recall(resp, groups):
    r = resp.lower()
    hits = [any(k.lower() in r for k in g) for g in groups]
    return sum(hits) / len(hits) if hits else None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# 顏色名稱或 hex 色碼都要能配對（Qwen2-VL 常輸出 "#FF0000" 這種格式）
COLOR_ALIASES = {"red": ["red", "紅", "#ff0000"], "blue": ["blue", "藍", "#0000ff"],
                 "green": ["green", "綠", "#008000", "#00ff00"]}


def parse_boxes(resp):
    """從回答中找出每個顏色後面出現的前 4 個數字作為 bbox；若座標皆 ≤1 視為 0~1 正規化並換算成 0~1000。"""
    boxes = {}
    for color, aliases in COLOR_ALIASES.items():
        for alias in aliases:
            m = re.search(re.escape(alias) + r"[^\d\[\(]{0,80}[\[\(]?\s*([\d.]+)\D+([\d.]+)\D+([\d.]+)\D+([\d.]+)", resp, re.I)
            if m:
                b = [float(x) for x in m.groups()]
                if max(b) <= 1.0:
                    b = [x * 1000 for x in b]
                boxes[color] = b
                break
    return boxes


def bbox_miou(resp, gt):
    pred = parse_boxes(resp)
    return sum(iou(pred[c], g) if c in pred else 0.0 for c, g in gt.items()) / len(gt)


# ----------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HW_DATA, "results"))
    ap.add_argument("--benchmark", default=os.path.join(HW_DATA, "vlm_benchmark.json"))
    ap.add_argument("--no-pairwise", action="store_true")
    ap.add_argument("--no-image", action="store_true", help="評審不看圖，只依參考答案評分（省錢）")
    args = ap.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        bench = {it["id"]: it for it in json.load(f)}
    with open(os.path.join(args.out_dir, "vlm_generations.jsonl"), encoding="utf-8") as f:
        gens = [json.loads(line) for line in f]
    models = list(dict.fromkeys(g["model"] for g in gens))

    # 1) 客觀指標 + pointwise 評分
    rows = []
    for g in gens:
        it = bench[g["id"]]
        img = None if args.no_image else os.path.join(HW_DATA, it["image"])
        row = {**g, "keyword_recall": keyword_recall(g["response"], it["keywords"])}
        if "gt_boxes" in it:
            row["bbox_mIoU"] = round(bbox_miou(g["response"], it["gt_boxes"]), 3)
        if it.get("hallucination_trap"):
            row["halluc_trap_pass"] = int(keyword_recall(g["response"], it["keywords"][-1:]) == 1.0)

        user = (f"【使用者指令】\n{it['prompt']}\n\n【參考答案】\n{it['reference']}\n\n"
                f"【本題評判維度】{'、'.join(it['judge_dims'])}\n\n【模型回答】\n{g['response']}")
        res = judge(POINTWISE_SYSTEM, user, image_path=img)
        for d in WEIGHTS:
            row[d] = res.get(d)
        if all(isinstance(row[d], (int, float)) for d in WEIGHTS):
            row["overall"] = round(sum(row[d] * w for d, w in WEIGHTS.items()), 2)
        row["hallucinated_items"] = "; ".join(res.get("hallucinated_items", []) or [])
        row["rationale"] = res.get("rationale", res.get("error"))
        rows.append(row)
        print(f"  {g['id']} | {g['model'].split('/')[-1]} | overall={row.get('overall')} kw={row['keyword_recall']:.2f}")

    df = pd.DataFrame(rows)
    df["judge_model"] = JUDGE_MODEL
    df.to_csv(os.path.join(args.out_dir, "vlm_scores.csv"), index=False, encoding="utf-8-sig")

    # 2) Pairwise（只在恰好兩個模型時執行）
    pair_df = None
    if not args.no_pairwise and len(models) == 2:
        A, B = models
        resp = {(g["id"], g["model"]): g["response"] for g in gens}
        prow = []
        for qid in dict.fromkeys(g["id"] for g in gens):
            it = bench[qid]
            img = None if args.no_image else os.path.join(HW_DATA, it["image"])
            verdicts = []
            for first, second in ((A, B), (B, A)):
                user = (f"【使用者指令】\n{it['prompt']}\n\n【參考答案】\n{it['reference']}\n\n"
                        f"【回答 A】\n{resp[(qid, first)]}\n\n【回答 B】\n{resp[(qid, second)]}")
                w = judge(PAIRWISE_SYSTEM, user, image_path=img).get("winner", "tie")
                verdicts.append({"A": first, "B": second}.get(w, "tie"))
            final = verdicts[0] if verdicts[0] == verdicts[1] else "tie"
            prow.append({"id": qid, "category": it["category"], "order1": verdicts[0], "order2": verdicts[1], "winner": final})
        pair_df = pd.DataFrame(prow)
        pair_df.to_csv(os.path.join(args.out_dir, "vlm_pairwise.csv"), index=False, encoding="utf-8-sig")

    # 3) 彙整與比較表
    metric_cols = [c for c in ["overall", *WEIGHTS, "keyword_recall", "bbox_mIoU", "halluc_trap_pass", "new_tokens", "latency_s"] if c in df]
    summary = df.groupby("model")[metric_cols].mean().round(3)
    by_cat = df.pivot_table(index="category", columns="model", values="overall", aggfunc="mean").round(2)
    if pair_df is not None:
        for m in models:
            summary.loc[m, "pairwise_win_rate"] = round((pair_df["winner"] == m).mean(), 3)
        summary["pairwise_tie_rate"] = round((pair_df["winner"] == "tie").mean(), 3)
    vram_path = os.path.join(args.out_dir, "vlm_vram.csv")
    if os.path.exists(vram_path):
        v = pd.read_csv(vram_path).set_index("model")
        summary["peak_vram_gib"] = (v["nvml_peak_used_mib"] / 1024).round(2)
    summary.to_csv(os.path.join(args.out_dir, "vlm_summary.csv"), encoding="utf-8-sig")
    by_cat.to_csv(os.path.join(args.out_dir, "vlm_by_category.csv"), encoding="utf-8-sig")

    with open(os.path.join(args.out_dir, "comparison_table.md"), "w", encoding="utf-8") as f:
        f.write(f"評審模型：{JUDGE_MODEL}\n\n## 總體指標\n\n")
        f.write(summary.T.rename(columns=lambda m: m.split("/")[-1]).to_markdown() + "\n\n")
        f.write("## 各類別 overall（1–10）\n\n")
        f.write(by_cat.rename(columns=lambda m: m.split("/")[-1]).to_markdown() + "\n")
        if pair_df is not None:
            f.write("\n## Pairwise 逐題勝負\n\n" + pair_df.to_markdown(index=False) + "\n")
    print("\n" + summary.T.to_string())
    print(f"\n輸出：vlm_scores.csv、vlm_summary.csv、vlm_by_category.csv、comparison_table.md 於 {args.out_dir}")


if __name__ == "__main__":
    main()
