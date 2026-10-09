"""Quiz 4 資料：Flickr8k（本機檔案）-> 切分、詞彙表、凍結 CNN / ViT 的影像特徵快取。

這是 Quiz 4 專用的新檔案，不會動到 data_loader.py。

預期資料放在 dataset/quiz4/ 底下（資料夾層數不限，程式會自動尋找）：
    圖片：  任一個放著很多 .jpg 的資料夾（例如 Images/ 或 Flicker8k_Dataset/）
    描述：  captions.txt / captions.csv      每行  image,caption（有標題列也可以）
            或 Flickr8k.token.txt            每行  image.jpg#0<Tab>caption
    切分：  若有 Flickr_8k.trainImages.txt / devImages.txt / testImages.txt 就用官方切分，
            否則以固定種子隨機切成 75% / 12.5% / 12.5%（與官方 6000/1000/1000 相近）

為了讓訓練很快，編碼器（ResNet50 或 ViT-B/16）凍結、只跑一次，抽出的特徵存成
dataset/quiz4/features_<encoder>_<split>.pt；之後訓練只用特徵，不必每個 epoch 重新讀圖。
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .data_loader import DATA, END, IMAGENET_MEAN, IMAGENET_STD, PAD, SEED, START, UNK, Vocab, tokenize

QUIZ4 = DATA / "quiz4"
IMG_EXT = {".jpg", ".jpeg", ".png"}
ENCODER_DIMS = {"resnet50": 2048, "vit_b_16": 768}
SPLIT_FILE = "caption_splits.json"


# ----------------------------------------------------------------------------
# 尋找並解析本機的 Flickr8k 檔案
# ----------------------------------------------------------------------------
def find_images_dir(root: Path = QUIZ4) -> Path:
    counts = Counter()
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            counts[p.parent] += 1
    if not counts:
        raise FileNotFoundError(f"在 {root} 底下找不到任何圖片（.jpg / .png）。請把 Flickr8k 的圖片放進這個資料夾。")
    return counts.most_common(1)[0][0]


def find_caption_file(root: Path = QUIZ4) -> Path:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".txt", ".csv")]
    by_name = {p.name.lower(): p for p in files}
    for name in ("captions.txt", "captions.csv", "flickr8k.token.txt"):
        if name in by_name:
            return by_name[name]
    for p in files:
        low = p.name.lower()
        if low.endswith(".token.txt") and "lemma" not in low:
            return p
    raise FileNotFoundError(
        f"在 {root} 底下找不到描述檔（captions.txt / captions.csv / Flickr8k.token.txt）。"
        f"目前找到的文字檔：{[p.name for p in files]}"
    )


def parse_captions(path: Path) -> dict:
    """回傳 {圖片檔名: [描述, ...]}。支援 'image,caption' 與 'image.jpg#0<Tab>caption' 兩種格式。"""
    caps = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            name, cap = line.split("\t", 1)
            name = name.split("#")[0]
        elif "," in line:
            name, cap = line.split(",", 1)  # 只切第一個逗號，描述裡可以有逗號
        else:
            continue
        name, cap = name.strip().strip('"'), cap.strip().strip('"')
        if Path(name).suffix.lower() not in IMG_EXT:  # 跳過標題列等非圖片名稱的行
            continue
        if cap:
            caps[name].append(cap)
    return dict(caps)


def _read_list(path: Path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_splits(root: Path = QUIZ4, seed: int = SEED, ratios=(0.75, 0.125)) -> dict:
    images_dir = find_images_dir(root)
    caps = parse_captions(find_caption_file(root))
    available = {p.name for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    names = sorted(n for n in caps if n in available)
    if not names:
        raise ValueError("描述檔裡的圖片名稱和圖片資料夾對不上，請確認兩者來自同一份 Flickr8k。")

    official = {
        "train": next(iter(root.rglob("Flickr_8k.trainImages.txt")), None),
        "val": next(iter(root.rglob("Flickr_8k.devImages.txt")), None),
        "test": next(iter(root.rglob("Flickr_8k.testImages.txt")), None),
    }
    if all(official.values()):
        source = "official"
        keep = set(names)
        parts = {s: [n for n in _read_list(p) if n in keep] for s, p in official.items()}
    else:
        source = f"random(seed={seed})"
        shuffled = names[:]
        random.Random(seed).shuffle(shuffled)
        n_train, n_val = int(len(shuffled) * ratios[0]), int(len(shuffled) * ratios[1])
        parts = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
    splits = {s: [{"image": n, "captions": caps[n]} for n in lst] for s, lst in parts.items()}
    rel = images_dir.relative_to(root).as_posix()
    return {"images_dir": rel, "source": source, **splits}


def load_splits(root: Path = QUIZ4):
    """回傳 (splits dict, images_dir Path)。切分只做一次並存成 caption_splits.json，之後固定沿用。"""
    path = root / SPLIT_FILE
    blob = None
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        if not (root / blob["images_dir"]).is_dir():
            blob = None  # 圖片資料夾被搬動過，重新建立
    if blob is None:
        blob = build_splits(root)
        path.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        print(f"已建立切分（{blob['source']}）並存到 {path.name}")
    splits = {s: blob[s] for s in ("train", "val", "test")}
    return splits, root / blob["images_dir"]


def describe_splits(splits):
    import pandas as pd

    rows = {}
    for s, recs in splits.items():
        lens = [len(tokenize(c)) for r in recs for c in r["captions"]]
        rows[s] = {
            "圖片數": len(recs),
            "描述句數": len(lens),
            "每張圖平均句數": round(len(lens) / len(recs), 2),
            "平均句長(字)": round(sum(lens) / len(lens), 2),
            "最長句(字)": max(lens),
        }
    return pd.DataFrame(rows).T


def build_caption_vocab(train_records, max_size=8000, min_freq=3) -> Vocab:
    counter = Counter()
    for r in train_records:
        for c in r["captions"]:
            counter.update(tokenize(c))
    return Vocab(counter, max_size, min_freq, specials=(PAD, UNK, START, END))  # <pad>=0, <unk>=1, <start>=2, <end>=3


# ----------------------------------------------------------------------------
# 影像編碼器（凍結）與特徵抽取
# ----------------------------------------------------------------------------
def eval_transform(size: int = 224):
    from torchvision import transforms as T

    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_encoder(name: str = "resnet50", pretrained: bool = True) -> nn.Module:
    """回傳凍結的編碼器；輸出為每張圖一條特徵向量（resnet50: 2048 維，vit_b_16: 768 維的 CLS）。"""
    from torchvision import models

    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Identity()
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads = nn.Identity()
    else:
        raise ValueError(f"encoder 需為 {list(ENCODER_DIMS)} 之一")
    for p in m.parameters():
        p.requires_grad = False
    return m.eval()


class _ImageFiles(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def extract_features(encoder, paths, device, batch_size=64, num_workers=2, size=224):
    loader = DataLoader(_ImageFiles(paths, eval_transform(size)), batch_size=batch_size, num_workers=num_workers)
    encoder.to(device).eval()
    out = [encoder(x.to(device)).float().cpu() for x in loader]
    return torch.cat(out)


def get_features(encoder_name, split, records, images_dir, device, pretrained=True, **kw):
    """取得某個切分的特徵（有快取就直接讀，沒有就抽取並存檔）。"""
    tag = encoder_name + ("" if pretrained else "_scratch")
    cache = QUIZ4 / f"features_{tag}_{split}.pt"
    names = [r["image"] for r in records]
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        if blob["images"] == names:
            return blob["feats"]
    print(f"抽取特徵 {encoder_name} / {split}：{len(names)} 張")
    enc = build_encoder(encoder_name, pretrained)
    feats = extract_features(enc, [images_dir / n for n in names], device, **kw)
    torch.save({"images": names, "feats": feats}, cache)
    return feats


# ----------------------------------------------------------------------------
# 訓練用 Dataset（特徵 + 描述）
# ----------------------------------------------------------------------------
class CaptionFeatureDataset(Dataset):
    """每個（圖片特徵, 一句描述）是一筆，所以一張圖會出現 5 次（5 句描述）。"""

    def __init__(self, feats, records, vocab, max_len=22):
        self.feats, self.items = feats, []
        start, end = vocab.stoi[START], vocab.stoi[END]
        for i, r in enumerate(records):
            for c in r["captions"]:
                ids = [start] + vocab.encode(tokenize(c))[:max_len] + [end]
                self.items.append((i, torch.tensor(ids)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, ids = self.items[k]
        return self.feats[i], ids


def caption_collate(batch):
    feats, ids = zip(*batch)
    return torch.stack(feats), pad_sequence(ids, batch_first=True, padding_value=0)  # <pad> 的 id 固定為 0


def prepare_caption_data(
    encoder_name="resnet50", device=None, batch_size=64, pretrained=True,
    max_vocab=8000, min_freq=3, max_len=22, num_workers=2,
):
    """一次準備好 Quiz 4 需要的全部資料。

    回傳 dict：splits、images_dir、vocab、feats（train/val/test 的特徵）、feat_dim、
    train_loader、val_loader（用來算驗證 loss）。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, images_dir = load_splits()
    vocab = build_caption_vocab(splits["train"], max_vocab, min_freq)  # 只用訓練集建立，避免資料洩漏
    feats = {
        s: get_features(encoder_name, s, splits[s], images_dir, device, pretrained, num_workers=num_workers)
        for s in ("train", "val", "test")
    }
    mk = lambda s: CaptionFeatureDataset(feats[s], splits[s], vocab, max_len)
    return dict(
        splits=splits,
        images_dir=images_dir,
        vocab=vocab,
        feats=feats,
        feat_dim=ENCODER_DIMS[encoder_name],
        train_loader=DataLoader(mk("train"), batch_size, shuffle=True, collate_fn=caption_collate),
        val_loader=DataLoader(mk("val"), batch_size, collate_fn=caption_collate),
    )
