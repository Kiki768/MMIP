"""
作業一：LLM 部署 × Prompting × 量化 × VRAM 量測 × LLM-as-a-Judge 一條龍腳本。

流程
  1. 依序以不同精度（fp16/bf16 / int8 / int4）載入模型，記錄載入後 VRAM。
  2. 對每種 Prompt 策略（zero_shot / few_shot / cot）批次推論 20 題，
     以 GPUMemoryMonitor 量測推論期間的平均與峰值 VRAM、延遲與 tokens/s。
  3. 呼叫商用 LLM 依 Rubric 評分（可用 --skip-judge 先只產生回答，之後 --judge-only 補評）。
  4. 輸出 generations.jsonl / judged.csv / vram.csv / summary_*.csv / report.json。

範例
  # 階段 1：FP16 下比較三種 Prompt 策略
  python src/hw1/run_llm_eval.py --precisions fp16 --strategies zero_shot few_shot cot --out-dir data/hw1/results/stage1
  # 階段 2：最佳策略（假設為 cot）× 量化精度，做最終評測
  python src/hw1/run_llm_eval.py --precisions fp16 int8 int4 --strategies cot --out-dir data/hw1/results/final
  # 只補跑評分
  python src/hw1/run_llm_eval.py --judge-only --out-dir data/hw1/results/final
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.gpu_monitor import GPUMemoryMonitor, free_gpu  # noqa: E402
from common.judge import MODEL as JUDGE_MODEL, judge  # noqa: E402
from common.paths import DATA  # noqa: E402

HW_DATA = os.path.join(DATA, "hw1")
SYSTEM_PROMPT = "你是一個知識豐富、樂於助人的 AI 助理。請一律使用繁體中文回答。"
COT_SUFFIX = "\n\n請先一步一步地推理（列出思考過程），最後另起一行以「最終答案：」開頭給出結論。"


# ----------------------------------------------------------------------------- 模型載入
def load_model(model_id, precision):
    """precision ∈ {fp16, bf16, int8, int4}。V100/T4 不支援 bf16，會自動退回 fp16。"""
    half = torch.float16
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            half = torch.bfloat16
        else:
            print("[警告] 此 GPU 不支援 bf16，改用 fp16")

    kwargs = {"device_map": "cuda:0", "torch_dtype": half}
    if precision == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4，對常態分佈權重較友善
            bnb_4bit_use_double_quant=True,     # 對量化常數再量化，再省約 0.4 bit/參數
            bnb_4bit_compute_dtype=half,        # 反量化後以 fp16/bf16 計算
        )

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"                   # decoder-only 批次生成必須左側 padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with GPUMemoryMonitor() as mon:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
    load_stats = mon.stats()
    load_stats["weights_footprint_mib"] = round(model.get_memory_footprint() / 1024 ** 2, 1)
    load_stats["after_load_alloc_mib"] = round(torch.cuda.memory_allocated() / 1024 ** 2, 1)
    return model, tok, load_stats


# ----------------------------------------------------------------------------- Prompt 策略
def build_messages(item, strategy, few_shot):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if strategy == "few_shot":
        for ex in few_shot.get(item["category"], []):
            msgs.append({"role": "user", "content": ex["q"]})
            msgs.append({"role": "assistant", "content": ex["a"]})
    prompt = item["prompt"] + (COT_SUFFIX if strategy == "cot" else "")
    msgs.append({"role": "user", "content": prompt})
    return msgs


@torch.inference_mode()
def generate_batch(model, tok, batch_msgs, max_new_tokens):
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,                        # greedy：讓不同精度的比較可重現
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    n_tokens = [int((g != tok.pad_token_id).sum()) for g in gen]
    return tok.batch_decode(gen, skip_special_tokens=True), n_tokens


def run_generation(args, bench):
    items = bench["items"][: args.limit] if args.limit else bench["items"]
    gens, vram_rows = [], []

    for precision in args.precisions:
        print(f"\n===== 載入 {args.model} @ {precision} =====")
        model, tok, load_stats = load_model(args.model, precision)
        print("載入 VRAM：", load_stats)

        for strategy in args.strategies:
            max_new = args.max_new_tokens * (2 if strategy == "cot" else 1)
            all_msgs = [build_messages(it, strategy, bench["few_shot_examples"]) for it in items]
            total_tokens = 0
            with GPUMemoryMonitor() as mon:
                for i in range(0, len(items), args.batch_size):
                    t0 = time.perf_counter()
                    outs, ntoks = generate_batch(model, tok, all_msgs[i:i + args.batch_size], max_new)
                    dt = time.perf_counter() - t0
                    total_tokens += sum(ntoks)
                    for it, o, n in zip(items[i:i + args.batch_size], outs, ntoks):
                        gens.append({"id": it["id"], "category": it["category"], "precision": precision,
                                     "strategy": strategy, "response": o.strip(), "new_tokens": n,
                                     "batch_latency_s": round(dt, 2)})
                    print(f"  [{precision}/{strategy}] {min(i + args.batch_size, len(items))}/{len(items)} 題 ({dt:.1f}s)")
            st = mon.stats()
            vram_rows.append({"precision": precision, "strategy": strategy,
                              **{f"load_{k}": v for k, v in load_stats.items()},
                              **{f"infer_{k}": v for k, v in st.items()},
                              "tokens_per_s": round(total_tokens / st["elapsed_s"], 1)})
            print("  推論 VRAM：", st)

        del model
        free_gpu()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "generations.jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    pd.DataFrame(vram_rows).to_csv(os.path.join(args.out_dir, "vram.csv"), index=False)
    return gens


# ----------------------------------------------------------------------------- LLM-as-a-Judge
JUDGE_SYSTEM = """你是一位嚴謹、公正的大型語言模型評審。你會看到【題目】、【參考答案】、【評分標準】與【模型回答】。
請依下列三個維度各給 1~10 的整數分數：

1. correctness（正確性）：事實、計算、推理與結論是否正確，是否符合參考答案與評分標準的核心要點。
   10=完全正確；7=大致正確有小瑕疵；4=部分正確但有關鍵錯誤；1=錯誤或答非所問。
2. instruction_following（指令遵循）：是否遵守題目要求的格式、字數、條數、角色設定與語言。
   使用簡體中文或英文回答（題目未要求時）最多給 6 分；超出字數/條數限制扣 2~4 分。
3. clarity（清晰度）：結構是否清楚、表達是否流暢、有無冗詞或重複。

評分規則：
- 不要因為回答較長就給高分；冗長重複應在 clarity 扣分。
- 回答中包含推理過程不扣分，但若「最終結論」錯誤，correctness 不得高於 3。
- 回答出現重複迴圈或被截斷而沒有結論，correctness 不得高於 4。
- 參考答案只是一種正確解法，其他等價且正確的答案應同樣給高分。

只輸出 JSON：{"correctness": int, "instruction_following": int, "clarity": int, "rationale": "50 字以內的扣分理由"}"""

WEIGHTS = {"correctness": 0.5, "instruction_following": 0.3, "clarity": 0.2}


def run_judge(args, bench, gens):
    by_id = {it["id"]: it for it in bench["items"]}
    rows = []
    for k, g in enumerate(gens, 1):
        it = by_id[g["id"]]
        user = (f"【題目】\n{it['prompt']}\n\n【參考答案】\n{it['reference']}\n\n"
                f"【評分標準】\n{it['criteria']}\n\n【模型回答】\n{g['response']}")
        res = judge(JUDGE_SYSTEM, user)
        row = {**g, **{d: res.get(d) for d in WEIGHTS}, "rationale": res.get("rationale", res.get("error"))}
        if all(isinstance(row[d], (int, float)) for d in WEIGHTS):
            row["overall"] = round(sum(row[d] * w for d, w in WEIGHTS.items()), 2)
        rows.append(row)
        print(f"  評分 {k}/{len(gens)} {g['id']} {g['precision']}/{g['strategy']} → {row.get('overall')}")

    df = pd.DataFrame(rows)
    df["judge_model"] = JUDGE_MODEL
    df.to_csv(os.path.join(args.out_dir, "judged.csv"), index=False, encoding="utf-8-sig")
    return df


def summarize(args, df):
    score_cols = list(WEIGHTS) + ["overall"]
    overall = df.groupby(["precision", "strategy"])[score_cols + ["new_tokens"]].mean().round(2).reset_index()
    by_cat = df.pivot_table(index=["precision", "strategy"], columns="category", values="overall", aggfunc="mean").round(2)
    overall.to_csv(os.path.join(args.out_dir, "summary_overall.csv"), index=False, encoding="utf-8-sig")
    by_cat.to_csv(os.path.join(args.out_dir, "summary_by_category.csv"), encoding="utf-8-sig")

    vram_path = os.path.join(args.out_dir, "vram.csv")
    vram = pd.read_csv(vram_path) if os.path.exists(vram_path) else pd.DataFrame()
    best = overall.sort_values("overall", ascending=False).iloc[0].to_dict()
    report = {"model": args.model, "judge_model": JUDGE_MODEL, "best_combination": best,
              "overall": overall.to_dict("records"), "by_category": by_cat.reset_index().to_dict("records"),
              "vram": vram.to_dict("records")}
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== 總分（依精度 × 策略）=====")
    print(overall.to_string(index=False))
    print("\n===== 各類別 overall =====")
    print(by_cat.to_string())
    print(f"\n最佳組合：{best['precision']} + {best['strategy']}（overall={best['overall']}）")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--benchmark", default=os.path.join(HW_DATA, "benchmark.json"))
    p.add_argument("--precisions", nargs="+", default=["fp16", "int8", "int4"], choices=["fp16", "bf16", "int8", "int4"])
    p.add_argument("--strategies", nargs="+", default=["zero_shot", "few_shot", "cot"], choices=["zero_shot", "few_shot", "cot"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512, help="CoT 自動加倍")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 題（除錯用）")
    p.add_argument("--out-dir", default=os.path.join(HW_DATA, "results"))
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--judge-only", action="store_true")
    args = p.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        bench = json.load(f)

    if args.judge_only:
        with open(os.path.join(args.out_dir, "generations.jsonl"), encoding="utf-8") as f:
            gens = [json.loads(line) for line in f]
    else:
        gens = run_generation(args, bench)

    if args.skip_judge:
        print(f"\n已輸出回答至 {args.out_dir}/generations.jsonl（未評分）")
        return
    summarize(args, run_judge(args, bench, gens))


if __name__ == "__main__":
    main()
