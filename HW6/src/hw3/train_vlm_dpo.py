"""
進階：以 DPO + LoRA 微調 Qwen2-VL，抑制幻覺（避免提及圖中不存在的物件）。

為什麼不用 trl.DPOTrainer？
  TRL 0.13 的 concatenated_forward 只轉傳 pixel_values / image_sizes，沒有 Qwen2-VL 必需的
  image_grid_thw，因此這裡直接實作相同的 sigmoid DPO 損失：
      loss = -log σ( β · [(log π(chosen) - log π_ref(chosen)) - (log π(rejected) - log π_ref(rejected))] )
  參考模型 π_ref 與 TRL 的 ref_model=None 做法相同：同一個模型「停用 LoRA adapter」。

防止資料洩漏
  前 --holdout-questions 題 POPE 用到的圖片（預設 300 題 = 50 張圖）完全不進訓練，留給 robustness_eval.py 評測。

可中斷續跑
  每 --save-every 個 optimizer step 存一次 adapter 與訓練狀態到 <out-dir>/checkpoint；
  重新執行同一個指令會自動接著訓練，已完成則直接結束。

用法：python src/hw3/train_vlm_dpo.py [--max-steps 2]（除錯用）
"""
import argparse
import json
import math
import os
import random
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import get_cosine_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "hw2"))
from common.paths import DATA  # noqa: E402
from robustness_eval import fetch  # noqa: E402
from run_vlm import load_vlm  # noqa: E402

COCO = os.path.join(DATA, "hw3", "coco")
# 只微調語言模型的線性層；視覺編碼器（visual.*）維持不動
LORA_TARGETS = r".*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def load_pairs(holdout_questions):
    with open(os.path.join(COCO, "pope_eval.jsonl"), encoding="utf-8") as f:
        pope = [json.loads(line) for line in f][:holdout_questions]
    heldout = {q["image_url"] for q in pope}
    with open(os.path.join(COCO, "halluc_dpo.jsonl"), encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    train = [p for p in pairs if p["images"][0] not in heldout]
    print(f"成對資料 {len(pairs)} 組 → 訓練 {len(train)} 組（排除 {len(heldout)} 張評測用圖片）")
    return train


def encode_pair(pair, processor, cache_dir):
    """把一組 (chosen, rejected) 編碼成 batch=2，並回傳 labels（prompt 與 padding 位置為 -100）。"""
    image = fetch(pair["images"][0], cache_dir)
    prompt = processor.apply_chat_template(pair["prompt"], add_generation_prompt=True)
    answers = [pair[k][0]["content"][0]["text"] + "<|im_end|>\n" for k in ("chosen", "rejected")]
    enc = processor(images=[image, image], text=[prompt + a for a in answers], padding=True, return_tensors="pt")
    prompt_len = processor(images=[image], text=[prompt], return_tensors="pt")["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[enc["attention_mask"] == 0] = -100
    return enc, labels


def answer_logprob(logits, labels):
    """每個序列在回答 token 上的 log 機率總和。"""
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    target = labels[:, 1:]
    mask = target != -100
    picked = logp.gather(2, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (picked * mask).sum(dim=1)


def plot_logs(logs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, keys in zip(axes, [["loss"], ["accuracy"], ["reward_chosen", "reward_rejected", "margin"]]):
        for k in keys:
            ax.plot([r["step"] for r in logs], [r[k] for r in logs], marker=".", label=k)
        ax.set_xlabel("optimizer step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("VLM-DPO training curves")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "vlm_dpo_training_curves.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--out-dir", default=os.path.join(DATA, "hw3", "adapters", "qwen2vl-dpo-halluc"))
    ap.add_argument("--holdout-questions", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8, help="每組資料是一個 batch，累積 8 組更新一次")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="只跑 N 個 optimizer step（除錯用）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    final_flag = os.path.join(args.out_dir, "adapter_model.safetensors")
    if os.path.exists(final_flag) and not args.max_steps:
        print(f"已完成訓練：{args.out_dir}")
        return

    ckpt = os.path.join(args.out_dir, "checkpoint")
    state_path = os.path.join(ckpt, "trainer_state.pt")
    resume = os.path.exists(state_path)
    cache_dir = os.path.join(COCO, "images")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = load_pairs(args.holdout_questions)
    base, processor = load_vlm(args.model)
    processor.tokenizer.padding_side = "right"
    if resume:
        model = PeftModel.from_pretrained(base, ckpt, is_trainable=True)
    else:
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
                         target_modules=LORA_TARGETS)
        model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()     # LoRA 權重用 float32 訓練，前向由 autocast 處理
    optim = torch.optim.AdamW(params, lr=args.lr)
    total_steps = args.epochs * len(pairs) // args.grad_accum
    sched = get_cosine_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)

    micro, logs = 0, []
    if resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        micro, logs = st["micro"], st["logs"]
        print(f"從 checkpoint 續跑：micro step {micro}，optimizer step {micro // args.grad_accum}/{total_steps}")

    dtype = next(p.dtype for p in base.parameters() if p.is_floating_point())
    # 每個 epoch 用固定種子洗牌，續跑時順序完全相同
    order = []
    for ep in range(args.epochs):
        idx = list(range(len(pairs)))
        random.Random(args.seed + ep).shuffle(idx)
        order += idx
    order = order[: total_steps * args.grad_accum]

    window = {"loss": 0.0, "acc": 0.0, "rc": 0.0, "rr": 0.0}
    for k in range(micro, len(order)):
        enc, labels = encode_pair(pairs[order[k]], processor, cache_dir)
        enc = {n: v.to(model.device) for n, v in enc.items()}
        labels = labels.to(model.device)
        with torch.autocast("cuda", dtype=dtype):
            with torch.no_grad(), model.disable_adapter():
                ref = answer_logprob(model(**enc).logits, labels)
            pi = answer_logprob(model(**enc).logits, labels)
            r_c, r_r = pi[0] - ref[0], pi[1] - ref[1]
            loss = -F.logsigmoid(args.beta * (r_c - r_r))
        (loss / args.grad_accum).backward()

        window["loss"] += loss.item()
        window["acc"] += float(pi[0] - pi[1] + (ref[1] - ref[0]) > 0)
        window["rc"] += r_c.item()
        window["rr"] += r_r.item()

        if (k + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step = (k + 1) // args.grad_accum
            n = args.grad_accum
            row = {"step": step, "loss": window["loss"] / n, "accuracy": window["acc"] / n,
                   "reward_chosen": window["rc"] / n, "reward_rejected": window["rr"] / n,
                   "margin": (window["rc"] - window["rr"]) / n}
            logs.append(row)
            window = dict.fromkeys(window, 0.0)
            print(f"step {step}/{total_steps} loss={row['loss']:.4f} acc={row['accuracy']:.2f} "
                  f"margin={row['margin']:.2f} peak={torch.cuda.max_memory_allocated() / 1024 ** 3:.1f}GiB", flush=True)

            if step % args.save_every == 0:
                tmp = ckpt + ".tmp"     # 先寫暫存再換名，存檔到一半被中斷也不會弄壞 checkpoint
                shutil.rmtree(tmp, ignore_errors=True)
                model.save_pretrained(tmp)
                torch.save({"optim": optim.state_dict(), "sched": sched.state_dict(), "micro": k + 1, "logs": logs},
                           os.path.join(tmp, "trainer_state.pt"))
                shutil.rmtree(ckpt, ignore_errors=True)
                os.replace(tmp, ckpt)
            if args.max_steps and step >= args.max_steps:
                print("除錯模式：提早結束（未寫入最終 adapter）")
                return

    model.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, "log_history.json"), "w") as f:
        json.dump(logs, f, indent=2)
    plot_logs(logs, args.out_dir)
    shutil.rmtree(ckpt, ignore_errors=True)
    print(f"LoRA adapter 與訓練曲線已存至 {args.out_dir}")
    print(f"峰值 VRAM：{torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB")


if __name__ == "__main__":
    main()
