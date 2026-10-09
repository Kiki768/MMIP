"""Quiz 1：三種資料集的下載、切分、前處理與 DataLoader。

資料位置（相對於專案根目錄）：
    dataset/quiz2  IMDB 50K 影評（文字分類）       -> 原始單一 CSV，自動切出 train/val/test.csv
    dataset/quiz3  CIFAR-10（影像分類，10 類）     -> torchvision 原始檔 + split_indices.json
    dataset/quiz4  Flickr8k（影像描述，每圖 5 句） -> images/*.jpg + captions_<split>.json
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"

PAD, UNK, START, END = "<pad>", "<unk>", "<start>", "<end>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# 共用：tokenizer 與 Vocab
# ----------------------------------------------------------------------------
def tokenize(text: str):
    text = re.sub(r"<br\s*/?>", " ", text.lower())
    return re.findall(r"[a-z0-9']+|[!?]", text)


class Vocab:
    def __init__(self, counter: Counter, max_size=20000, min_freq=2, specials=(PAD, UNK)):
        self.itos = list(specials)
        for word, freq in counter.most_common():
            if freq < min_freq or len(self.itos) >= max_size:
                break
            if word not in specials:
                self.itos.append(word)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids, skip_special=True):
        words = [self.itos[i] for i in ids]
        if skip_special:
            words = [w for w in words if w not in (PAD, START, END)]
        return " ".join(words)


def build_vocab(texts, max_size=20000, min_freq=2, specials=(PAD, UNK)):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    return Vocab(counter, max_size, min_freq, specials)


# ----------------------------------------------------------------------------
# Quiz 1-1：文字分類資料集（IMDB 50K，單一 CSV）
# ----------------------------------------------------------------------------
_TEXT_COLS = ("review", "text")
_LABEL_COLS = ("sentiment", "label")


def _find_source_csv(folder: Path) -> Path:
    """在 dataset/quiz2 中找原始的單一 CSV（排除程式自己產生的 train/val/test.csv）。"""
    for path in sorted(folder.glob("*.csv")):
        if path.stem in ("train", "val", "test"):
            continue
        cols = pd.read_csv(path, nrows=1).columns
        if any(c in cols for c in _TEXT_COLS) and any(c in cols for c in _LABEL_COLS):
            return path
    raise FileNotFoundError(
        f"在 {folder} 找不到原始資料 CSV（需包含 review/sentiment 欄位）。請把 IMDB 的 CSV 放進這個資料夾。"
    )


def prepare_text(val_ratio: float = 0.1, test_ratio: float = 0.1):
    """讀取單一 IMDB CSV，整理欄位後分層切成 train / val / test（預設 80 / 10 / 10）。

    - 欄位統一成 text、label（positive -> 1、negative -> 0）
    - 先去除重複評論，避免同一則評論同時出現在訓練集與測試集
    - 以 SEED 固定、依標籤分層切分，三份 CSV 存回 dataset/quiz2（已存在則略過）
    """
    from sklearn.model_selection import train_test_split

    out = DATA / "quiz2"
    out.mkdir(parents=True, exist_ok=True)
    if all((out / f"{s}.csv").exists() for s in ("train", "val", "test")):
        return

    raw = pd.read_csv(_find_source_csv(out))
    text_col = next(c for c in _TEXT_COLS if c in raw.columns)
    label_col = next(c for c in _LABEL_COLS if c in raw.columns)
    df = raw[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna()

    if not pd.api.types.is_numeric_dtype(df["label"]):  # 文字標籤（positive / negative）轉成 1 / 0
        df["label"] = df["label"].astype(str).str.strip().str.lower().map({"positive": 1, "negative": 0})
        if df["label"].isna().any():
            raise ValueError("sentiment 欄位只能是 positive / negative")
    df["label"] = df["label"].astype(int)

    n_before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"原始 {n_before} 筆，去除重複評論後 {len(df)} 筆")

    rest, test_df = train_test_split(df, test_size=test_ratio, stratify=df["label"], random_state=SEED)
    train_df, val_df = train_test_split(
        rest, test_size=val_ratio / (1 - test_ratio), stratify=rest["label"], random_state=SEED
    )
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.reset_index(drop=True).to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} 筆（positive {int(part['label'].sum())} / negative {int((part['label'] == 0).sum())}）")


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, max_len: int = 256):
        self.labels = df["label"].tolist()
        self.ids = []
        for text in df["text"]:
            ids = vocab.encode(tokenize(text))[:max_len]
            ids += [vocab.pad_id] * (max_len - len(ids))
            self.ids.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.tensor(self.ids[i]), torch.tensor(self.labels[i])


def load_text_data(batch_size=64, max_len=256, max_vocab=20000):
    """回傳 (train_loader, val_loader, test_loader, vocab)。"""
    prepare_text()
    out = DATA / "quiz2"
    train_df, val_df, test_df = (pd.read_csv(out / f"{s}.csv") for s in ("train", "val", "test"))
    vocab = build_vocab(train_df["text"], max_size=max_vocab)  # 只用訓練集建詞彙表，避免資料洩漏
    mk = lambda df: TextDataset(df, vocab, max_len)
    return (
        DataLoader(mk(train_df), batch_size, shuffle=True),
        DataLoader(mk(val_df), batch_size),
        DataLoader(mk(test_df), batch_size),
        vocab,
    )


# ----------------------------------------------------------------------------
# Quiz 1-2：影像分類資料集（CIFAR-10，ImageFolder 格式）
# ----------------------------------------------------------------------------
# 預期的資料夾結構：
#   dataset/quiz3/train/<class_name>/*.png
#   dataset/quiz3/test/<class_name>/*.png      （類別資料夾名稱與 train 相同）
def prepare_vision(val_ratio: float = 0.1):
    """檢查資料夾結構，並從 train 固定切出驗證集（每類各取 val_ratio）。

    切分索引存成 split_indices.json，之後 ViT / ResNet 都用同一份切分，比較才公平。
    test 資料夾完全不參與訓練與挑選模型，只在最後評估時使用。
    """
    from torchvision.datasets import ImageFolder

    out = DATA / "quiz3"
    for split in ("train", "test"):
        if not (out / split).is_dir():
            raise FileNotFoundError(f"找不到 {out / split}，請確認資料夾內有 train/ 與 test/，且各含類別子資料夾。")

    labels = np.array(ImageFolder(out / "train").targets)  # 只列出檔案，不會讀取圖片內容
    split_file = out / "split_indices.json"
    if split_file.exists():
        with open(split_file) as f:
            split = json.load(f)
        if len(split["train"]) + len(split["val"]) == len(labels):
            return  # 已有且與目前的 train 資料夾筆數相符，沿用同一份切分

    rng = np.random.RandomState(SEED)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n_val = int(len(idx) * val_ratio)
        val_idx += idx[:n_val].tolist()
        train_idx += idx[n_val:].tolist()
    with open(split_file, "w") as f:
        json.dump({"train": sorted(train_idx), "val": sorted(val_idx)}, f)


def cifar_transforms(size: int = 224):
    """CIFAR-10 原圖很小（32x32），先放大到預訓練模型需要的 size（224），再做輕量增強。

    增強（隨機裁切 + 水平翻轉）只用在訓練；驗證與測試只做放大與標準化。
    """
    from torchvision import transforms as T

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # 預訓練權重使用 ImageNet 統計量
    train_tf = T.Compose(
        [
            T.Resize((size, size)),
            T.RandomCrop(size, padding=size // 8),  # 相當於在 32x32 原圖上補 4 像素後裁切
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
        ]
    )
    eval_tf = T.Compose([T.Resize((size, size)), T.ToTensor(), norm])
    return train_tf, eval_tf


def vision_transforms(size: int = 224):
    """一般照片用（Flickr8k captioning 使用）。"""
    from torchvision import transforms as T

    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = T.Compose([T.RandomResizedCrop(size), T.RandomHorizontalFlip(), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(), norm])
    return train_tf, eval_tf


def load_vision_data(batch_size=32, size=224, num_workers=2):
    """回傳 (train_loader, val_loader, test_loader, class_names)。"""
    from torch.utils.data import Subset
    from torchvision.datasets import ImageFolder

    prepare_vision()
    out = DATA / "quiz3"
    with open(out / "split_indices.json") as f:
        split = json.load(f)
    train_tf, eval_tf = cifar_transforms(size)
    train_full = ImageFolder(out / "train", transform=train_tf)
    val_full = ImageFolder(out / "train", transform=eval_tf)  # 同一批原始圖片，驗證時不做增強
    test_ds = ImageFolder(out / "test", transform=eval_tf)
    if train_full.classes != test_ds.classes:
        raise ValueError(f"train 與 test 的類別資料夾不一致：{train_full.classes} vs {test_ds.classes}")
    train_ds = Subset(train_full, split["train"])
    val_ds = Subset(val_full, split["val"])
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, **kw),
        DataLoader(test_ds, **kw),
        train_full.classes,  # 依資料夾名稱字母排序，標籤 0..9 與此順序對應
    )


# ----------------------------------------------------------------------------
# Quiz 1-3：影像 - 文字描述資料集（Flickr8k）
# ----------------------------------------------------------------------------
def prepare_caption():
    """下載 Flickr8k（官方切分 6000/1000/1000），存圖片與 captions json。"""
    out = DATA / "quiz4"
    img_dir = out / "images"
    if (out / "captions_train.json").exists():
        return
    img_dir.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    ds = load_dataset("jxie/flickr8k")
    for name, key in (("train", "train"), ("val", "validation"), ("test", "test")):
        records = []
        for i, ex in enumerate(ds[key]):
            fname = f"{name}_{i:05d}.jpg"
            ex["image"].convert("RGB").save(img_dir / fname, quality=95)
            caps = [ex[k] for k in sorted(ex) if k.startswith("caption")]
            records.append({"image": fname, "captions": caps})
        with open(out / f"captions_{name}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)


def load_caption_records(split: str):
    with open(DATA / "quiz4" / f"captions_{split}.json", encoding="utf-8") as f:
        return json.load(f)


def build_caption_vocab(records, max_size=8000, min_freq=3):
    counter = Counter()
    for r in records:
        for c in r["captions"]:
            counter.update(tokenize(c))
    return Vocab(counter, max_size, min_freq, specials=(PAD, UNK, START, END))


class CaptionDataset(Dataset):
    """train 模式：每個 (圖, 句) 一筆，共約 6000x5 筆；
    eval 模式：每張圖一筆，refs 保留全部 5 句供 BLEU / BERTScore 使用。"""

    def __init__(self, records, vocab, transform, mode="train"):
        from PIL import Image

        self._open = Image.open
        self.img_dir = DATA / "quiz4" / "images"
        self.vocab, self.transform = vocab, transform
        if mode == "train":
            self.items = [(r["image"], c, r["captions"]) for r in records for c in r["captions"]]
        else:
            self.items = [(r["image"], r["captions"][0], r["captions"]) for r in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, cap, refs = self.items[i]
        img = self.transform(self._open(self.img_dir / fname).convert("RGB"))
        v = self.vocab
        ids = [v.stoi[START]] + v.encode(tokenize(cap)) + [v.stoi[END]]
        return img, torch.tensor(ids), refs


def caption_collate(batch):
    imgs, caps, refs = zip(*batch)
    lengths = torch.tensor([len(c) for c in caps])
    padded = pad_sequence(caps, batch_first=True, padding_value=0)  # <pad> 的 id 固定為 0
    return torch.stack(imgs), padded, lengths, list(refs)


def load_caption_data(batch_size=32, size=224, num_workers=2):
    """回傳 (train_loader, val_loader, test_loader, vocab)。"""
    prepare_caption()
    train_tf, eval_tf = vision_transforms(size)
    recs = {s: load_caption_records(s) for s in ("train", "val", "test")}
    vocab = build_caption_vocab(recs["train"])
    kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=caption_collate)
    return (
        DataLoader(CaptionDataset(recs["train"], vocab, train_tf, "train"), shuffle=True, **kw),
        DataLoader(CaptionDataset(recs["val"], vocab, eval_tf, "eval"), **kw),
        DataLoader(CaptionDataset(recs["test"], vocab, eval_tf, "eval"), **kw),
        vocab,
    )
