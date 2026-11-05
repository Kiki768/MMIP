"""
作業二：兩個開源 VLM 在相同 Prompt 下的推論腳本（Qwen2-VL vs LLaVA-NeXT）。

  - 兩個模型「依序」載入，每跑完一個就釋放顯存，因此 24GB 卡也能以 fp16 跑兩個 7B 模型；
    16GB 卡請加 --load-4bit。
  - 統一使用 greedy decoding、相同 max_new_tokens、相同 Prompt（無 system prompt），確保公平。
  - 記錄每題延遲、輸出 token 數，以及每個模型推論期間的平均/峰值 VRAM。

用法
  python src/hw2/make_synthetic_images.py        # 先準備圖片
  python src/hw2/run_vlm.py                      # 預設 Qwen2-VL-7B-Instruct 與 LLaVA-v1.6-Mistral-7B
  python src/hw2/run_vlm.py --models Qwen/Qwen2-VL-2B-Instruct --limit 3   # 快速除錯
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.gpu_monitor import GPUMemoryMonitor, free_gpu  # noqa: E402
from common.paths import DATA  # noqa: E402

HW_DATA = os.path.join(DATA, "hw2")
DEFAULT_MODELS = ["Qwen/Qwen2-VL-7B-Instruct", "llava-hf/llava-v1.6-mistral-7b-hf"]


def load_vlm(model_id, load_4bit=False):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"torch_dtype": dtype, "device_map": "cuda:0"}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

    proc_kwargs = {}
    if "qwen2" in model_id.lower():
        # Qwen2-VL 採動態解析度：圖片切成 28x28 patch，限制 patch 數以控制顯存與速度
        proc_kwargs = {"min_pixels": 256 * 28 * 28, "max_pixels": 1024 * 28 * 28}
    processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
    model = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs).eval()
    return model, processor


@torch.inference_mode()
def vlm_generate(model, processor, image, prompt, max_new_tokens):
    # Qwen2-VL 與 LLaVA-NeXT 的 processor 都支援這種「多模態訊息」格式的 chat template
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=[image], text=[text], return_tensors="pt").to(model.device)
    # 兩個模型使用完全相同的解碼設定；輕微 repetition_penalty 抑制小模型的重複迴圈
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05,
                         temperature=None, top_p=None, top_k=None, pad_token_id=processor.tokenizer.pad_token_id
                         or processor.tokenizer.eos_token_id)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip(), int(gen.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--benchmark", default=os.path.join(HW_DATA, "vlm_benchmark.json"))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--load-4bit", action="store_true", help="16GB 顯卡請開啟")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(HW_DATA, "results"))
    args = ap.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)

    gens, vram_rows = [], []
    for model_id in args.models:
        print(f"\n===== 載入 {model_id} =====")
        model, processor = load_vlm(model_id, args.load_4bit)
        weights_mib = round(model.get_memory_footprint() / 1024 ** 2, 1)

        with GPUMemoryMonitor() as mon:
            for it in items:
                image = Image.open(os.path.join(HW_DATA, it["image"])).convert("RGB")
                t0 = time.perf_counter()
                resp, ntok = vlm_generate(model, processor, image, it["prompt"], args.max_new_tokens)
                dt = time.perf_counter() - t0
                gens.append({"id": it["id"], "category": it["category"], "model": model_id,
                             "response": resp, "new_tokens": ntok, "latency_s": round(dt, 2)})
                print(f"  [{it['id']}] {dt:.1f}s | {resp[:60]!r}")
        vram_rows.append({"model": model_id, "weights_mib": weights_mib, **mon.stats()})

        del model, processor
        free_gpu()

    with open(os.path.join(args.out_dir, "vlm_generations.jsonl"), "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    pd.DataFrame(vram_rows).to_csv(os.path.join(args.out_dir, "vlm_vram.csv"), index=False)
    print(f"\n輸出：{args.out_dir}/vlm_generations.jsonl、vlm_vram.csv")


if __name__ == "__main__":
    main()
