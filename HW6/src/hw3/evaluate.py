"""
作業三：微調前（base）vs 微調後（base + DPO LoRA）在兩個測試集上的量化比較。

只載入一次模型：
  - 「base」以 `with model.disable_adapter():` 停用 LoRA 來評估
  - 「dpo」 正常啟用 LoRA
兩者共用同一份權重與相同的 greedy 解碼設定，比較最公平。

A. 目標能力（data/hw3/target_test.jsonl，30 題未見過主題）
   pure_trad_rate  純繁體率 ↑      simp_char_ratio 簡體字比例 ↓
   cjk_ratio       中文字比例（防止改用英文規避）  avg_new_tokens 平均長度
   judge_quality   （選用 --judge）商用 LLM 評「內容品質」1~10，刻意忽略字體，檢查內容有沒有變差
B. 通用能力（data/hw3/general_bench.jsonl，100 題 MMLU）
   mmlu_acc：比較 A/B/C/D 四個選項 token 的 logit（不需生成，穩定且快）
   McNemar 精確檢定：判斷微調前後正確率差異是否顯著

用法：python src/hw3/evaluate.py --adapter data/hw3/adapters/qwen-dpo-zhtw [--judge]
"""
import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from common.paths import DATA  # noqa: E402
from zh_metrics import zh_metrics  # noqa: E402

HW_DATA = os.path.join(DATA, "hw3")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.inference_mode()
def generate_all(model, tok, prompts, max_new_tokens, batch_size=8):
    outs, ntoks = [], []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        gen = gen[:, enc["input_ids"].shape[1]:]
        outs += [o.strip() for o in tok.batch_decode(gen, skip_special_tokens=True)]
        ntoks += [int((g != tok.pad_token_id).sum()) for g in gen]
    return outs, ntoks


@torch.inference_mode()
def mmlu_predict(model, tok, items):
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in "ABCD"]
    preds = []
    for it in items:
        q = it["question"] + "\n" + "\n".join(f"{c}. {t}" for c, t in zip("ABCD", it["choices"]))
        q += "\nAnswer with only the letter (A, B, C, or D)."
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
        logits = model(**enc).logits[0, -1, letter_ids]
        preds.append("ABCD"[int(logits.argmax())])
    return preds


def mcnemar_p(base_correct, dpo_correct):
    """精確 McNemar 檢定（雙尾二項檢定）：只看兩者判定不一致的題目。"""
    b = sum(1 for x, y in zip(base_correct, dpo_correct) if x and not y)
    c = sum(1 for x, y in zip(base_correct, dpo_correct) if y and not x)
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), b, c


QUALITY_SYSTEM = """你是評審。請評估【回答】對【問題】的內容品質（正確性、完整度、條理），給 1~10 整數。
注意：完全忽略繁體/簡體字的差異，只看內容本身。只輸出 JSON：{"score": int, "reason": "30 字以內"}"""


def evaluate(model, tok, tag, target, general, args):
    print(f"\n===== 評估 {tag} =====")
    prompts = [t["prompt"] for t in target]
    outs, ntoks = generate_all(model, tok, prompts, args.max_new_tokens)
    rows = [{"model": tag, "prompt": p, "response": o, "new_tokens": n, **zh_metrics(o)}
            for p, o, n in zip(prompts, outs, ntoks)]
    if args.judge:
        from common.judge import judge
        for r in rows:
            r["judge_quality"] = judge(QUALITY_SYSTEM, f"【問題】\n{r['prompt']}\n\n【回答】\n{r['response']}").get("score")
    preds = mmlu_predict(model, tok, general)
    correct = [p == it["answer"] for p, it in zip(preds, general)]
    print(f"  純繁體率={sum(r['is_pure_trad'] for r in rows) / len(rows):.2%}  MMLU={sum(correct) / len(correct):.2%}")
    return rows, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=os.path.join(HW_DATA, "adapters", "qwen-dpo-zhtw"))
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--judge", action="store_true", help="以商用 LLM 評內容品質（需 API key）")
    ap.add_argument("--out-dir", default=os.path.join(HW_DATA, "results", "eval_dpo"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    target = read_jsonl(os.path.join(HW_DATA, "target_test.jsonl"))
    general = read_jsonl(os.path.join(HW_DATA, "general_bench.jsonl"))

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="cuda:0")
    model = PeftModel.from_pretrained(base, args.adapter).eval()

    with model.disable_adapter():
        base_rows, base_correct = evaluate(model, tok, "base", target, general, args)
    dpo_rows, dpo_correct = evaluate(model, tok, "dpo", target, general, args)

    df = pd.DataFrame(base_rows + dpo_rows)
    df.to_csv(os.path.join(args.out_dir, "target_responses.csv"), index=False, encoding="utf-8-sig")

    agg = {"pure_trad_rate": ("is_pure_trad", "mean"), "simp_char_ratio": ("simp_char_ratio", "mean"),
           "cjk_ratio": ("cjk_ratio", "mean"), "avg_new_tokens": ("new_tokens", "mean")}
    if args.judge:
        agg["judge_quality"] = ("judge_quality", "mean")
    summary = df.groupby("model", sort=False).agg(**agg)
    summary["mmlu_acc"] = [sum(base_correct) / len(base_correct), sum(dpo_correct) / len(dpo_correct)]
    summary = summary.round(4)
    summary.loc["delta (dpo-base)"] = summary.loc["dpo"] - summary.loc["base"]
    p, b, c = mcnemar_p(base_correct, dpo_correct)

    summary.to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), encoding="utf-8-sig")
    report = {"summary": summary.reset_index().to_dict("records"),
              "mmlu_mcnemar": {"base_only_correct": b, "dpo_only_correct": c, "p_value": round(p, 4)},
              "mmlu_per_subject": pd.DataFrame({"subject": [g["subject"] for g in general],
                                                "base": base_correct, "dpo": dpo_correct}).to_dict("records")}
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 圖：微調前後指標對照
    cols = [c for c in ["pure_trad_rate", "simp_char_ratio", "cjk_ratio", "mmlu_acc"] if c in summary]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = range(len(cols))
    for k, (tag, color) in enumerate([("base", "#8C8C8C"), ("dpo", "#4C72B0")]):
        bars = ax.bar([i + (k - 0.5) * 0.38 for i in x], summary.loc[tag, cols], 0.38, label=tag, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(list(x), cols)
    ax.set_ylim(0, 1.1)
    ax.set_title("Before vs after DPO")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "before_after.png"), dpi=150)

    print("\n" + summary.to_string())
    print(f"\nMMLU McNemar：只有 base 答對 {b} 題、只有 dpo 答對 {c} 題，p = {p:.4f}"
          f"（{'顯著' if p < 0.05 else '不顯著'}，α=0.05）")


if __name__ == "__main__":
    main()
