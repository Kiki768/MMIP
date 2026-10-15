# -*- coding: utf-8 -*-
"""Quiz 2：多模態分類（Hateful Memes）。

影像端（ResNet-50 / ViT-B/16）+ 文字端（BERT / RoBERTa 的 [CLS]）→ Concatenation → MLP 分類頭。

資料切分說明：
    - train.jsonl 內部再切 90% / 10% 為 train / val（val 只用來挑選最佳 epoch）。
    - dev.jsonl 當作最終的 Test 集（官方 test 標籤未公開），避免在測試集上挑 epoch。
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .utils import (
    build_image_backbone,
    build_imagenet_transform,
    classification_metrics,
    count_parameters,
    ensure_dir,
    get_logger,
    plot_training_curves,
    save_table,
    set_seed,
)

logger = get_logger()

# 組合代號 -> 設定
COMBOS: Dict[str, dict] = { # 模型實驗組合
    "A": {"name": "A: ResNet-50 + BERT", "image": "resnet50", "text": "bert-base-uncased"},
    "B": {"name": "B: ViT-B/16 + BERT", "image": "vit_b_16", "text": "bert-base-uncased"},
    "C": {"name": "C: ResNet-50 + RoBERTa", "image": "resnet50", "text": "roberta-base"},
}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> List[dict]:
    """讀取 jsonl，忽略空行；單行格式錯誤時給出行號。"""
    if not Path(path).is_file():
        raise FileNotFoundError(f"找不到檔案：{path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {ln} 行不是合法 JSON：{e}") from e
    return records


class HatefulMemesDataset(Dataset): # 自訂 Dataset
    """同時回傳影像張量、BERT/RoBERTa 的 input_ids / attention_mask 與標籤。"""

    def __init__(self, records: List[dict], img_root: Path, tokenizer, transform, max_len: int = 64):
        self.records = records
        self.img_root = Path(img_root)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.missing = 0  # 統計讀不到的影像數量

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str) -> Image.Image:
        try:
            return Image.open(self.img_root / rel_path).convert("RGB")
        except Exception as e:  # 檔案遺失 / 損毀：以黑圖替代，避免訓練中斷
            self.missing += 1
            if self.missing <= 3:
                logger.warning("影像讀取失敗，改用黑圖：%s (%s)", rel_path, e)
            return Image.new("RGB", (224, 224))

    def __getitem__(self, idx: int) -> dict: # 回傳單筆資料
        r = self.records[idx]
        image = self.transform(self._load_image(r["img"]))
        enc = self.tokenizer(
            r.get("text", ""),
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "image": image,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(r.get("label", -1)), dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class MultimodalClassifier(nn.Module):
    """Concatenation 融合 + MLP 分類頭。凍結的骨幹固定在 eval 模式並以 no_grad 前向。"""

    def __init__(
        self,
        image_name: str,
        text_name: str,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        num_classes: int = 2,
        freeze_image: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel

        self.image_encoder, img_dim = build_image_backbone(image_name)
        self.text_encoder = AutoModel.from_pretrained(text_name)
        txt_dim = self.text_encoder.config.hidden_size
        self.freeze_image, self.freeze_text = freeze_image, freeze_text
        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        fused = img_dim + txt_dim
        self.classifier = nn.Sequential( # 把多個神經網路層依序串起來。輸入傳入第一層，輸出再傳給下一層。
            nn.LayerNorm(fused),
            nn.Linear(fused, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # 凍結的骨幹不應啟用 Dropout / BatchNorm 更新
        if self.freeze_image:
            self.image_encoder.eval()
        if self.freeze_text:
            self.text_encoder.eval()
        return self

    def forward(self, image, input_ids, attention_mask):
        if self.freeze_image:
            with torch.no_grad():
                img_feat = self.image_encoder(image)
        else:
            img_feat = self.image_encoder(image)

        def _text():
            return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

        if self.freeze_text:
            with torch.no_grad():
                txt_feat = _text()  # [CLS]（RoBERTa 為 <s>）向量
        else:
            txt_feat = _text()
        return self.classifier(torch.cat([img_feat.float(), txt_feat.float()], dim=1))


# --------------------------------------------------------------------------- #
# 訓練 / 評估
# --------------------------------------------------------------------------- #
def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    model.eval()
    losses, preds, probs, labels = [], [], [], []
    use_amp = device.type == "cuda"
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(batch["image"], batch["input_ids"], batch["attention_mask"])
        losses.append(criterion(logits.float(), batch["label"]).item() * len(batch["label"]))
        probs.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(batch["label"].cpu().tolist())
    out = classification_metrics(labels, preds, probs)
    out["loss"] = sum(losses) / max(len(labels), 1)
    return out


def _trainable_state(model: nn.Module) -> dict:
    """只備份可訓練參數，避免複製整個凍結骨幹佔用大量記憶體。"""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def train_one_combo(
    combo_key: str,
    records: dict,
    img_root: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr_head: float,
    lr_backbone: float,
    finetune: bool,
    num_workers: int,
    max_len: int,
    seed: int,
    quick: bool,
    result_dir: Optional[Path] = None,
):
    """訓練並評估單一組合，回傳 (summary_dict, history_df)。"""
    from transformers import AutoTokenizer

    cfg = COMBOS[combo_key]
    logger.info("===== 組合 %s =====", cfg["name"])
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg["text"])
    transform = build_imagenet_transform(224)
    make = lambda recs: HatefulMemesDataset(recs, img_root, tokenizer, transform, max_len)  # noqa: E731

    train_recs, val_recs, test_recs = records["train"], records["val"], records["test"]
    train_ds, val_ds, test_ds = make(train_recs), make(val_recs), make(test_recs)
    pin = device.type == "cuda"
    mk_loader = lambda ds, shuffle: DataLoader(  # noqa: E731
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin,
        drop_last=False,
    )
    train_loader, val_loader, test_loader = mk_loader(train_ds, True), mk_loader(val_ds, False), mk_loader(test_ds, False)

    model = MultimodalClassifier(
        cfg["image"], cfg["text"], freeze_image=not finetune, freeze_text=not finetune
    ).to(device)
    n_train = count_parameters(model, trainable_only=True)
    logger.info("可訓練參數：%.2fM / 總參數：%.2fM", n_train / 1e6, count_parameters(model) / 1e6)

    # 類別不平衡（non-hateful 約 64%）→ 以反比例權重緩解
    lab = np.array([int(r["label"]) for r in train_recs])
    counts = np.bincount(lab, minlength=2).astype(float)
    weights = torch.tensor(len(lab) / (2.0 * np.maximum(counts, 1)), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    head_params = list(model.classifier.parameters())
    groups = [{"params": head_params, "lr": lr_head}]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.")]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    history, best_f1, best_epoch, best_state = [], -1.0, 0, None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(batch["image"], batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits.float(), batch["label"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * len(batch["label"])
            seen += len(batch["label"])

        val = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": run_loss / max(seen, 1),
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_f1": val["f1"],
        }
        history.append(row)
        logger.info(
            "epoch %d | train_loss %.4f | val_loss %.4f | val_acc %.2f | val_f1 %.2f",
            epoch, row["train_loss"], row["val_loss"], row["val_acc"], row["val_f1"],
        )
        if val["f1"] > best_f1:
            best_f1, best_epoch, best_state = val["f1"], epoch, _trainable_state(model)
    train_time = time.time() - t0

    # 以驗證集 F1 最佳的 epoch 參數，在 Test（dev.jsonl）上做最終評估
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        if result_dir is not None:  # 存下最佳權重，供之後的單筆推論展示載入
            torch.save(
                {"trainable_state": best_state, "combo": combo_key, "finetune": finetune, "max_len": max_len},
                Path(result_dir) / f"best_combo_{combo_key}.pt",
            )
    test = evaluate(model, test_loader, device, criterion)
    logger.info("Test | acc %.2f | f1 %.2f | macro-f1 %.2f | auroc %.2f", test["accuracy"], test["f1"], test["macro_f1"], test["auroc"])

    summary = {
        "Combination": cfg["name"],
        "Image Backbone": cfg["image"],
        "Text Backbone": cfg["text"],
        "Mode": "end-to-end finetune" if finetune else "frozen backbones",
        "Trainable Params (M)": round(n_train / 1e6, 2),
        "Best Epoch": best_epoch,
        "Val F1 (%)": round(best_f1, 2),
        "Test Accuracy (%)": round(test["accuracy"], 2),
        "Test F1 (%)": round(test["f1"], 2),
        "Test Macro-F1 (%)": round(test["macro_f1"], 2),
        "Test AUROC (%)": round(test["auroc"], 2),
        "Train Time (s)": round(train_time, 1),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, pd.DataFrame(history)


def run(
    data_dir: Path,
    result_dir: Path,
    device: torch.device,
    seed: int = 42,
    combos: Optional[List[str]] = None,
    epochs: int = 5,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_backbone: float = 2e-5,
    finetune: bool = False,
    num_workers: int = 2,
    max_len: int = 64,
    val_ratio: float = 0.1,
    quick: bool = False,
) -> pd.DataFrame:
    """執行 Quiz 2，回傳兩（或多）組合的比較表。"""
    data_dir, result_dir = Path(data_dir), ensure_dir(result_dir)
    combos = [c.upper() for c in (combos or ["A", "B"])]
    bad = [c for c in combos if c not in COMBOS]
    if bad:
        raise ValueError(f"未知的組合代號 {bad}，可用：{list(COMBOS)}")

    img_root = data_dir
    if not (data_dir / "img").is_dir():
        raise FileNotFoundError(f"找不到影像資料夾：{data_dir / 'img'}")
    train_all = read_jsonl(data_dir / "train.jsonl")
    test_recs = read_jsonl(data_dir / "dev.jsonl")
    

    # train → train / val
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(train_all))
    n_val = max(1, int(len(train_all) * val_ratio))
    val_recs = [train_all[i] for i in perm[:n_val]]
    train_recs = [train_all[i] for i in perm[n_val:]]
    if quick:  # 冒煙測試：極小子集 + 1 epoch
        train_recs, val_recs, test_recs = train_recs[:128], val_recs[:64], test_recs[:64]
        epochs = 1
    logger.info("資料量：train=%d, val=%d, test(dev)=%d", len(train_recs), len(val_recs), len(test_recs))
    records = {"train": train_recs, "val": val_recs, "test": test_recs}

    summaries, histories, failures = [], {}, []
    for key in combos:
        try:
            s, h = train_one_combo(
                key, records, img_root, device, epochs, batch_size, lr_head, lr_backbone,
                finetune, num_workers, max_len, seed, quick, result_dir,
            )
            summaries.append(s)
            histories[COMBOS[key]["name"]] = h
            h.to_csv(result_dir / f"history_combo_{key}.csv", index=False)
        except Exception as e:  # 單一組合失敗不影響其他組合
            logger.exception("組合 %s 執行失敗：%s", key, e)
            failures.append(key)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not summaries:
        raise RuntimeError(f"Quiz 2 所有組合皆失敗：{failures}")
    if histories:
        plot_training_curves(histories, result_dir / "training_curves.png")
    df = pd.DataFrame(summaries)
    save_table(df, result_dir / "metrics.csv")
    if failures:
        logger.warning("以下組合失敗而未列入表格：%s", failures)

    # 訓練完自動做單筆推論展示（quick 模式資料太少，跳過）
    if not quick:
        for key in combos:
            if key in failures:
                continue
            try:
                predict_samples(key, data_dir, result_dir, device)
            except Exception as e:  # 展示失敗不影響已產生的指標表格
                logger.error("組合 %s 的推論展示失敗：%s", key, e)
    return df


# --------------------------------------------------------------------------- #
# 單筆推論展示：從 dev.jsonl 挑樣本，顯示 圖片 / 文字 / 預測 / 真實標籤
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_samples(
    combo_key: str,
    data_dir: Path,
    result_dir: Path,
    device: torch.device,
    n_per_class: int = 3,
    seed: int = 7,
):
    """載入訓練好的組合，對 dev.jsonl 內挑出的樣本做推論並畫圖。

    每個類別（hateful / non-hateful）各挑 n_per_class 筆，所以圖上會有答對也可能有答錯。
    回傳 DataFrame（id、文字、真實標籤、預測、hateful 機率、是否答對），並存成圖片與 CSV。
    """
    import matplotlib.pyplot as plt
    import textwrap
    from transformers import AutoTokenizer

    data_dir, result_dir = Path(data_dir), Path(result_dir)
    ckpt_path = result_dir / f"best_combo_{combo_key}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到權重 {ckpt_path}，請先執行 Quiz 2 訓練（會自動存檔）。")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = COMBOS[combo_key]

    model = MultimodalClassifier(
        cfg["image"], cfg["text"], freeze_image=not ckpt["finetune"], freeze_text=not ckpt["finetune"]
    )
    model.load_state_dict(ckpt["trainable_state"], strict=False)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg["text"])
    transform = build_imagenet_transform(224)

    # 各類別隨機挑樣本
    records = read_jsonl(data_dir / "dev.jsonl")
    rng = np.random.RandomState(seed)
    picked = []
    for lab in (1, 0):
        pool = [r for r in records if int(r["label"]) == lab]
        picked += [pool[i] for i in rng.choice(len(pool), min(n_per_class, len(pool)), replace=False)]

    rows, images = [], []
    for r in picked:
        img = Image.open(data_dir / r["img"]).convert("RGB")
        enc = tokenizer(r["text"], max_length=ckpt["max_len"], truncation=True,
                        padding="max_length", return_tensors="pt")
        logits = model(
            transform(img).unsqueeze(0).to(device),
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
        prob = torch.softmax(logits.float(), dim=1)[0].cpu()
        pred = int(prob.argmax())
        images.append(img)
        rows.append({
            "id": r["id"], "text": r["text"], "true_label": int(r["label"]),
            "pred_label": pred, "hateful_prob": round(float(prob[1]), 4), "correct": pred == int(r["label"]),
        })

    # 畫圖：綠框 = 預測正確，紅框 = 預測錯誤
    names = {0: "non-hateful", 1: "hateful"}
    ncols = 3
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 5.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (img, row) in enumerate(zip(images, rows)):
        ax = axes[i // ncols][i % ncols]
        ax.axis("on")
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        color = "tab:green" if row["correct"] else "tab:red"
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(4)
        text = "\n".join(textwrap.wrap("Text: " + row["text"], 46))
        ax.set_title(
            f"True: {names[row['true_label']]}  |  Pred: {names[row['pred_label']]}"
            f" (P(hateful)={row['hateful_prob']:.2f})\n{text}",
            fontsize=9, loc="left",
        )
    fig.suptitle(f"Multimodal inference samples - {cfg['name']} (green=correct, red=wrong)", fontsize=12)
    fig.tight_layout()
    fig.savefig(result_dir / f"prediction_samples_{combo_key}.png", dpi=150, bbox_inches="tight")
    plt.show()

    df = pd.DataFrame(rows)
    df.to_csv(result_dir / f"prediction_samples_{combo_key}.csv", index=False, encoding="utf-8-sig")
    save_table(df, result_dir / f"prediction_samples_{combo_key}.csv")
    return df