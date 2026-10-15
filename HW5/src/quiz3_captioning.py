# -*- coding: utf-8 -*-
"""Quiz 3：CLIP Image Captioning（ClipCap 風格）。

架構：凍結的 CLIP Vision Encoder → MLP Mapping Network → Prefix Embeddings → GPT-2 Decoder
訓練：訓練 Mapping Network 並微調 GPT-2（Cross-Entropy，僅對文字 token 計算 loss）。
評估：BLEU-1~4（nltk）與 BERTScore，並與 Week 4 Baseline 對照。
"""
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import (
    count_parameters,
    ensure_dir,
    get_logger,
    plot_caption_samples,
    save_table,
    set_seed,
)

logger = get_logger()

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------- #
# 資料處理
# --------------------------------------------------------------------------- #
def find_image_dir(base: Path) -> Path:
    """找出實際放有 .jpg 的資料夾（容許多包一層子資料夾，如官方的 Flicker8k_Dataset）。"""
    base = Path(base)
    if not base.is_dir():
        raise FileNotFoundError(f"找不到影像資料夾：{base}")
    for cand in [base] + sorted(p for p in base.iterdir() if p.is_dir()):
        if next(cand.glob("*.jpg"), None) is not None:
            return cand
    raise FileNotFoundError(f"{base} 內找不到任何 .jpg 影像")


def load_captions(token_file: Path) -> Dict[str, List[str]]:
    """解析 Flickr8k.token.txt：每行 `xxx.jpg#0<TAB>caption`，同一張圖 5 句。"""
    if not Path(token_file).is_file():
        raise FileNotFoundError(f"找不到標註檔：{token_file}")
    caps: Dict[str, List[str]] = {}
    with open(token_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
            if len(parts) < 2:
                continue
            img_id = parts[0].split("#")[0].strip()
            caps.setdefault(img_id, []).append(parts[1].strip().lower())
    return caps


def load_split_ids(text_dir: Path, name: str) -> Optional[List[str]]:
    """讀取官方切分清單（train/dev/test）；找不到則回傳 None。"""
    for p in sorted(Path(text_dir).rglob(f"*{name}Images.txt")):
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return None


def build_splits(
    text_dir: Path, image_dir: Path, caps: Dict[str, List[str]], seed: int
) -> Tuple[List[str], List[str], List[str]]:
    """取得 train/val/test 影像 id；官方清單不存在時退回 6000/1000/1000 隨機切分。"""
    exist = {p.name for p in image_dir.glob("*.jpg")}
    valid = lambda ids: [i for i in ids if i in exist and i in caps]  # noqa: E731
    tr, va, te = (load_split_ids(text_dir, n) for n in ("train", "dev", "test"))
    if tr and va and te:
        splits = valid(tr), valid(va), valid(te)
    else:
        logger.warning("找不到官方切分清單，改用隨機 6000/1000/1000 切分。")
        ids = sorted(valid(list(caps.keys())))
        random.Random(seed).shuffle(ids)
        splits = ids[:6000], ids[6000:7000], ids[7000:8000]
    if not all(splits):
        raise RuntimeError("切分後有空的集合，請確認 Flickr8k 影像與標註檔是否對應。")
    return splits


class Flickr8kDataset(Dataset):
    """Flickr8k Dataset。

    flatten=True ：每筆為 (影像, 其中一句 caption)，供訓練 / 驗證 loss 使用。
    flatten=False：每張影像一筆，只回傳影像張量與索引，供推論生成使用。
    """

    def __init__(self, image_dir, captions, image_ids, transform, tokenizer=None, max_len=40, flatten=True):
        self.image_dir = Path(image_dir)
        self.transform, self.tokenizer, self.max_len, self.flatten = transform, tokenizer, max_len, flatten
        self.image_ids = list(image_ids)
        if flatten:
            self.samples = [(i, c) for i in self.image_ids for c in captions[i]]
        else:
            self.samples = [(i, None) for i in self.image_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_id, cap = self.samples[idx]
        pixel = self.transform(Image.open(self.image_dir / img_id).convert("RGB"))
        if not self.flatten:
            return {"pixel_values": pixel, "index": idx}
        enc = self.tokenizer(
            cap + self.tokenizer.eos_token,  # 以 <|endoftext|> 作為句尾
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def build_clip_transform(size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class MappingNetwork(nn.Module):
    """MLP 投影層：CLIP 影像向量 (clip_dim) → prefix_len 個 GPT-2 embedding (gpt_dim)。"""

    def __init__(self, clip_dim: int, gpt_dim: int, prefix_len: int):
        super().__init__()
        self.prefix_len, self.gpt_dim = prefix_len, gpt_dim
        hidden = (gpt_dim * prefix_len) // 2
        self.mlp = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, gpt_dim * prefix_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).view(-1, self.prefix_len, self.gpt_dim)


class ClipCaptionModel(nn.Module):
    """CLIP（凍結）+ Mapping Network + GPT-2。"""

    def __init__(self, clip_name: str = "openai/clip-vit-base-patch32", gpt_name: str = "gpt2", prefix_len: int = 10):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection, GPT2LMHeadModel

        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_name)
        for p in self.clip.parameters():  # 凍結 CLIP
            p.requires_grad = False
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt_name)
        clip_dim = self.clip.config.projection_dim
        gpt_dim = self.gpt2.config.n_embd
        self.prefix_len = prefix_len
        self.mapper = MappingNetwork(clip_dim, gpt_dim, prefix_len)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # CLIP 永遠保持 eval
        return self

    def encode_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            img_emb = self.clip(pixel_values=pixel_values).image_embeds  # [B, 512]
        return self.mapper(img_emb.float())  # [B, P, 768]

    def forward(self, pixel_values, input_ids, attention_mask):
        B = input_ids.size(0)
        prefix = self.encode_prefix(pixel_values)
        tok_emb = self.gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix.to(tok_emb.dtype), tok_emb], dim=1)
        attn = torch.cat(
            [torch.ones(B, self.prefix_len, dtype=attention_mask.dtype, device=attention_mask.device), attention_mask],
            dim=1,
        )
        # prefix 與 padding 位置不計 loss（-100）
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        labels = torch.cat([torch.full((B, self.prefix_len), -100, dtype=labels.dtype, device=labels.device), labels], dim=1)
        return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels).loss

    @torch.no_grad()
    def generate(self, pixel_values, eos_id: int, max_new_tokens: int = 30, num_beams: int = 3) -> torch.Tensor:
        """以影像 prefix 啟動 GPT-2 生成（Beam Search）。"""
        prefix = self.encode_prefix(pixel_values)
        attn = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
        return self.gpt2.generate(
            inputs_embeds=prefix,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )


@torch.no_grad()
def generate_captions(model, tokenizer, loader, device, max_new_tokens=30, num_beams=3) -> List[str]:
    """對 DataLoader（flatten=False）內所有影像生成描述。"""
    model.eval()
    outs: List[str] = []
    use_amp = device.type == "cuda"
    for batch in tqdm(loader, desc="generate", leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            ids = model.generate(
                batch["pixel_values"].to(device), tokenizer.eos_token_id, max_new_tokens, num_beams
            )
        outs.extend(t.strip() or "a" for t in tokenizer.batch_decode(ids, skip_special_tokens=True))
    return outs


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    use_amp = device.type == "cuda"
    for b in tqdm(loader, desc="val", leave=False):
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = model(b["pixel_values"], b["input_ids"], b["attention_mask"])
        tot += loss.item() * len(b["input_ids"])
        n += len(b["input_ids"])
    return tot / max(n, 1)


# --------------------------------------------------------------------------- #
# 評估指標
# --------------------------------------------------------------------------- #
def _tok(s: str) -> List[str]:
    """簡單且不需下載 punkt 的斷詞：小寫英數字與縮寫。"""
    return re.findall(r"[a-z0-9']+", s.lower())


def compute_bleu(hyps: List[str], refs: List[List[str]]) -> Dict[str, float]:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    h = [_tok(x) for x in hyps]
    r = [[_tok(x) for x in rs] for rs in refs]
    sm = SmoothingFunction().method1
    weights = {
        "BLEU-1": (1, 0, 0, 0),
        "BLEU-2": (0.5, 0.5, 0, 0),
        "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25),
    }
    return {k: corpus_bleu(r, h, weights=w, smoothing_function=sm) for k, w in weights.items()}


def compute_bertscore(hyps: List[str], refs: List[List[str]], device: torch.device) -> Dict[str, float]:
    """BERTScore（多參考句時取最高分）。失敗（如無法下載模型）時回傳 NaN 而不中斷整個流程。"""
    try:
        from bert_score import score

        P, R, F = score(hyps, refs, lang="en", device=str(device), batch_size=64, verbose=False)
        return {"BERTScore-P": P.mean().item(), "BERTScore-R": R.mean().item(), "BERTScore-F1": F.mean().item()}
    except Exception as e:
        logger.error("BERTScore 計算失敗：%s", e)
        return {"BERTScore-P": float("nan"), "BERTScore-R": float("nan"), "BERTScore-F1": float("nan")}


def load_week4_baseline(path: Path) -> Dict[str, Optional[float]]:
    """讀取 Week 4 Baseline 指標（JSON）。不存在時產生範本讓使用者填入數值。"""
    keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BERTScore-P", "BERTScore-R", "BERTScore-F1"]
    path = Path(path)
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: data.get(k) for k in keys}
        except Exception as e:
            logger.error("讀取 Week 4 baseline 失敗（%s），欄位將顯示 N/A", e)
    else:
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: None for k in keys}, f, ensure_ascii=False, indent=2)
        logger.warning("已建立 Week 4 baseline 範本：%s（請填入你 Week 4 的數值後重跑評估）", path)
    return {k: None for k in keys}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(
    data_dir: Path,
    result_dir: Path,
    device: torch.device,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 32,
    lr_mapper: float = 1e-4,
    lr_gpt: float = 2e-5,
    prefix_len: int = 10,
    max_len: int = 40,
    num_workers: int = 2,
    max_train_images: Optional[int] = None,
    num_val_images: int = 1000,
    num_eval_images: int = 1000,
    num_beams: int = 3,
    num_vis: int = 6,
    week4_baseline: Optional[Path] = None,
    quick: bool = False,
) -> pd.DataFrame:
    """執行 Quiz 3：訓練 → 生成 → 指標表格 → 範例圖。"""
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    data_dir, result_dir = Path(data_dir), ensure_dir(result_dir)
    set_seed(seed)

    # 1) 資料
    image_dir = find_image_dir(data_dir / "Flickr8k_Dataset")
    text_dir = data_dir / "Flickr8k_text"
    token_file = next(iter(sorted(text_dir.rglob("Flickr8k.token.txt"))), text_dir / "Flickr8k.token.txt")
    caps = load_captions(token_file)
    train_ids, val_ids, test_ids = build_splits(text_dir, image_dir, caps, seed)
    if max_train_images:
        train_ids = train_ids[:max_train_images]
    val_ids = val_ids[:num_val_images]
    test_ids = test_ids[:num_eval_images]
    if quick:
        train_ids, val_ids, test_ids, epochs = train_ids[:64], val_ids[:16], test_ids[:16], 1
    logger.info("影像數：train=%d val=%d test=%d", len(train_ids), len(val_ids), len(test_ids))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    transform = build_clip_transform(224)
    pin = device.type == "cuda"
    mk = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin
    )
    train_loader = mk(Flickr8kDataset(image_dir, caps, train_ids, transform, tokenizer, max_len, True), True)
    val_loader = mk(Flickr8kDataset(image_dir, caps, val_ids, transform, tokenizer, max_len, True), False)
    test_ds = Flickr8kDataset(image_dir, caps, test_ids, transform, flatten=False)
    test_loader = mk(test_ds, False)

    # 2) 模型與優化器
    model = ClipCaptionModel(prefix_len=prefix_len).to(device)
    logger.info("可訓練參數：%.2fM（CLIP 已凍結）", count_parameters(model, True) / 1e6)
    optimizer = torch.optim.AdamW(
        [{"params": model.mapper.parameters(), "lr": lr_mapper}, {"params": model.gpt2.parameters(), "lr": lr_gpt}],
        weight_decay=1e-2,
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    # 3) 訓練（以驗證 loss 挑選最佳權重並存檔）
    ckpt = result_dir / "clipcap_best.pt"
    best_val, history = float("inf"), []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for b in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(b["pixel_values"], b["input_ids"], b["attention_mask"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * len(b["input_ids"])
            n += len(b["input_ids"])
        v = eval_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": run_loss / max(n, 1), "val_loss": v})
        logger.info("epoch %d | train_loss %.4f | val_loss %.4f", epoch, run_loss / max(n, 1), v)
        if v < best_val:
            best_val = v
            torch.save({"mapper": model.mapper.state_dict(), "gpt2": model.gpt2.state_dict(), "prefix_len": prefix_len}, ckpt)
    train_time = time.time() - t0
    pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)
    if ckpt.is_file():  # 載回最佳 epoch
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.mapper.load_state_dict(state["mapper"])
        model.gpt2.load_state_dict(state["gpt2"])

    # 4) 推論生成 + 評估
    hyps = generate_captions(model, tokenizer, test_loader, device, num_beams=num_beams)
    refs = [caps[i] for i in test_ids]
    bleu = compute_bleu(hyps, refs)
    bert = compute_bertscore(hyps, refs, device)
    ours = {**bleu, **bert}

    pd.DataFrame(
        {"image": test_ids, "generated": hyps, "references": [" || ".join(r) for r in refs]}
    ).to_csv(result_dir / "generated_captions.csv", index=False, encoding="utf-8-sig")

    # 5) 指標表格（含 Week 4 Baseline 對照）
    baseline = load_week4_baseline(week4_baseline or (result_dir / "week4_baseline.json"))
    rows = []
    for k, v in ours.items():
        b = baseline.get(k)
        rows.append(
            {
                "Metric": k,
                "Week 4 Baseline": round(b, 4) if isinstance(b, (int, float)) else "N/A",
                "CLIP+GPT-2 (Ours)": round(v, 4) if v == v else "N/A",
                "Improvement (Δ)": round(v - b, 4) if isinstance(b, (int, float)) and v == v else "N/A",
            }
        )
    df = pd.DataFrame(rows)
    save_table(df, result_dir / "metrics.csv")
    logger.info("Quiz 3 訓練耗時 %.1fs，評估影像 %d 張。", train_time, len(test_ids))

    # 6) 範例圖
    pick = np.random.RandomState(seed).choice(len(test_ids), size=min(num_vis, len(test_ids)), replace=False)
    samples = [
        {
            "image": Image.open(image_dir / test_ids[i]).convert("RGB"),
            "generated": hyps[i],
            "reference": refs[i][0],
        }
        for i in pick
    ]
    plot_caption_samples(samples, result_dir / "caption_samples.png")
    return df


@torch.no_grad()
def caption_single_image(model: ClipCaptionModel, tokenizer, image_path: str, device: torch.device, num_beams: int = 3) -> str:
    """推論函式：輸入一張圖片路徑，輸出文字描述。"""
    model.eval()
    pixel = build_clip_transform(224)(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    ids = model.generate(pixel, tokenizer.eos_token_id, num_beams=num_beams)
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()
