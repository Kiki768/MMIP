"""
自動合成作業三所需的三份資料：

  data/hw3/dpo_train.jsonl      100 組 DPO 成對資料（TRL 對話格式：prompt / chosen / rejected）
  data/hw3/target_test.jsonl     30 題「未見過主題」的繁中遵循度測試 prompt（與訓練集主題不重疊）
  data/hw3/general_bench.jsonl  100 題 MMLU 多選題（跨學科抽樣），用來檢查通用能力是否退化

成對資料的合成策略（「同內容、只差字體」）
  1. 用基礎模型對每個繁中 prompt 生成回答（基礎模型常混入簡體字，這正是要修正的行為）。
  2. chosen   = OpenCC s2twp(回答) → 臺灣正體＋臺灣慣用詞（如「软件」→「軟體」）
     rejected = OpenCC t2s(回答)   → 完整簡體
  chosen 與 rejected 的「內容、長度、推理」完全相同，唯一差異就是字體/用詞，
  因此 DPO 的梯度訊號只會推動「繁體偏好」，最大程度降低對通用能力的干擾。

用法：python src/hw3/build_datasets.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import os
import random
import sys

import torch
from datasets import load_dataset
from opencc import OpenCC
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import DATA as _DATA  # noqa: E402

DATA = os.path.join(_DATA, "hw3")

TEMPLATES = [
    "請用三句話介紹{t}。",
    "{t}有哪些常見的誤解？請列出三點並說明。",
    "請寫一段 150 字左右的短文，主題是「{t}」。",
    "如果要向國小學生解釋{t}，你會怎麼說？",
    "請列出學習{t}的入門步驟。",
    "{t}對日常生活有什麼影響？請舉例說明。",
    "請比較{t}的優點與缺點。",
    "請為一場關於{t}的十分鐘演講擬定大綱。",
    "請針對{t}出三道選擇題，並附上答案。",
    "用輕鬆幽默的語氣跟我聊聊{t}。",
]
TRAIN_TOPICS = ["人工智慧", "氣候變遷", "咖啡文化", "太空探索", "個人理財",
                "健康飲食", "網路安全", "電動車", "古典音樂", "時間管理"]
TEST_TOPICS = ["再生能源", "寵物照顧", "城市交通"]   # 測試主題不出現在訓練集 → 檢驗泛化


def make_prompts(topics):
    return [tpl.format(t=t) for t in topics for tpl in TEMPLATES]


@torch.inference_mode()
def generate(model, tok, prompts, max_new_tokens, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        # 取樣解碼讓回答更多樣；top_p/temperature 適中避免胡言亂語
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  生成 {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return [o.strip() for o in outs]


def build_general_bench(n, seed):
    """從 MMLU test split 跨 57 個學科均勻抽樣 n 題。"""
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    by_subj = {}
    for i, s in enumerate(ds["subject"]):
        by_subj.setdefault(s, []).append(i)
    subjects = sorted(by_subj)
    rng.shuffle(subjects)
    picked = [rng.choice(by_subj[subjects[k % len(subjects)]]) for k in range(n)]
    return [{"subject": ds[i]["subject"], "question": ds[i]["question"],
             "choices": ds[i]["choices"], "answer": "ABCD"[ds[i]["answer"]]} for i in picked]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"寫入 {len(rows)} 筆 → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(DATA, exist_ok=True)

    train_prompts = make_prompts(TRAIN_TOPICS)                 # 10 主題 × 10 模板 = 100
    test_prompts = make_prompts(TEST_TOPICS)                   # 3 主題 × 10 模板 = 30

    s2twp, t2s = OpenCC("s2twp"), OpenCC("t2s")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="cuda:0").eval()
    responses = generate(model, tok, train_prompts, args.max_new_tokens, args.batch_size)
    # 若回答幾乎是英文，繁簡轉換後 chosen == rejected，沒有訓練訊號 → 重新生成（最多 3 輪）
    for _ in range(3):
        bad = [i for i, r in enumerate(responses) if s2twp.convert(r) == t2s.convert(r)]
        if not bad:
            break
        print(f"  {len(bad)} 筆回答無繁簡差異，重新生成")
        for i, r in zip(bad, generate(model, tok, [train_prompts[i] for i in bad], args.max_new_tokens, args.batch_size)):
            responses[i] = r
    pairs =[{"prompt": [{"role": "user", "content": p}],
              "chosen": [{"role": "assistant", "content": s2twp.convert(r)}],
              "rejected": [{"role": "assistant", "content": t2s.convert(r)}]}
             for p, r in zip(train_prompts, responses)]
    write_jsonl(os.path.join(DATA, "dpo_train.jsonl"), pairs)
    write_jsonl(os.path.join(DATA, "target_test.jsonl"), [{"prompt": p} for p in test_prompts])
    write_jsonl(os.path.join(DATA, "general_bench.jsonl"), build_general_bench(100, args.seed))


if __name__ == "__main__":
    main()
