# -*- coding: utf-8 -*-
"""Quiz 1：影像檢索（Image Retrieval）。

流程：
    本地 CIFAR-10（dataset/quiz1/ 內的 train / test）
    → Query 取自 test、Gallery 取自 train（兩者天然不重疊）
    → ResNet-50 與 ViT-B/16 抽特徵 → L2 Normalization → Cosine Similarity
    → Top-5 檢索 → 指標表格 + 視覺化

支援的本地資料版面（自動偵測，不會下載）：
    (1) 影像資料夾：dataset/quiz1/{train,test}/<類別資料夾>/*.png|jpg
    (2) 官方 python 版：dataset/quiz1/cifar-10-batches-py/（data_batch_1~5、test_batch）
    (3) 以上兩者被多包一層資料夾（例如 dataset/quiz1/cifar10/train/...）也會自動往下找
"""
import pickle
import time
import gc
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import (
    build_image_backbone,
    build_imagenet_transform,
    count_parameters,
    ensure_dir,
    get_logger,
    plot_retrieval_samples,
    save_table,
    set_seed,
    topk_retrieval_metrics,
)

logger = get_logger()

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 對比模型：顯示名稱 -> backbone 代號
BACKBONES = {
    "ResNet-50 (CNN)": "resnet50",
    "ViT-B/16 (Transformer)": "vit_b_16",
}


# --------------------------------------------------------------------------- #
# 本地資料來源（統一介面：len / targets / get_image(i)）
# --------------------------------------------------------------------------- #
class ArraySource:
    """記憶體中的 uint8 陣列 (N,32,32,3)，來自官方 cifar-10-batches-py。"""

    def __init__(self, images: np.ndarray, targets: np.ndarray):
        self.images, self.targets = images, np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def get_image(self, i: int) -> np.ndarray:
        return self.images[i]


class FileSource:
    """影像檔案清單，讀取時才開檔（不一次載入記憶體）。"""

    def __init__(self, paths: List[Path], targets: np.ndarray):
        self.paths, self.targets = paths, np.asarray(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.targets)

    def get_image(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(self.paths[i]).convert("RGB"))


def _find_dir(root: Path, names: Sequence[str]) -> Optional[Path]:
    """在 root 底下（含一層子資料夾）找名稱符合 names（不分大小寫）的資料夾。"""
    root = Path(root)
    for base in [root] + sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(base.iterdir()):
            if p.is_dir() and p.name.lower() in names:
                return p
    return None


def _scan_image_folder(split_dir: Path, class_names: Optional[List[str]]):
    """掃描 <split>/<class>/*.png 結構；class_names 給定時沿用同一個類別順序。"""
    if class_names is None:
        class_names = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    cls_idx = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    for c in class_names:
        d = split_dir / c
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in IMG_EXTS:
                paths.append(f)
                labels.append(cls_idx[c])
    if not paths:
        raise FileNotFoundError(f"{split_dir} 內找不到「類別資料夾/影像檔」結構")
    return FileSource(paths, np.array(labels)), class_names


def _load_batches_py(root: Path):
    """讀取官方 cifar-10-batches-py（不呼叫 torchvision 下載）。"""
    d = next((p for p in [root, *root.rglob("cifar-10-batches-py")] if (p / "test_batch").is_file()), None)
    if d is None:
        return None

    def read(fn):
        with open(d / fn, "rb") as f:
            b = pickle.load(f, encoding="latin1")
        x = np.asarray(b["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return x, np.asarray(b["labels"], dtype=np.int64)

    names = CIFAR10_CLASSES
    meta = d / "batches.meta"
    if meta.is_file():
        with open(meta, "rb") as f:
            names = list(pickle.load(f, encoding="latin1").get("label_names", names))
    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6) if (d / f"data_batch_{i}").is_file()])
    train = ArraySource(np.concatenate(xs), np.concatenate(ys)) if xs else None
    test = ArraySource(*read("test_batch"))
    return train, test, names


def load_local_cifar10(data_root: Path, allow_download: bool = False):
    """載入本地 CIFAR-10，回傳 (train_source, test_source, class_names)。

    allow_download=False（預設）時絕不連網；找不到資料就丟出清楚的錯誤訊息。
    """
    data_root = Path(data_root)
    ensure_dir(data_root)

    # (1) 影像資料夾版面
    train_dir = _find_dir(data_root, {"train", "training"})
    test_dir = _find_dir(data_root, {"test", "testing", "val", "valid", "validation"})
    if test_dir is not None:
        test_src, names = _scan_image_folder(test_dir, None)
        train_src = _scan_image_folder(train_dir, names)[0] if train_dir is not None else None
        # 類別資料夾若是 0~9 數字，換成 CIFAR-10 的英文名稱方便閱讀圖表
        if len(names) == 10 and all(n.isdigit() for n in names):
            names = [CIFAR10_CLASSES[int(n)] for n in names]
        logger.info("偵測到影像資料夾版面：train=%s test=%s", train_dir, test_dir)
        return train_src, test_src, names

    # (2) 官方 python 版
    got = _load_batches_py(data_root)
    if got is not None:
        logger.info("偵測到 cifar-10-batches-py 版面")
        return got

    # (3) 明確允許時才下載
    if allow_download:
        from torchvision.datasets import CIFAR10

        tr = CIFAR10(root=str(data_root), train=True, download=True)
        te = CIFAR10(root=str(data_root), train=False, download=True)
        return ArraySource(tr.data, tr.targets), ArraySource(te.data, te.targets), CIFAR10_CLASSES

    tree = sorted(p.name for p in data_root.iterdir())[:20]
    raise FileNotFoundError(
        f"在 {data_root} 找不到 CIFAR-10。需要 train/ 與 test/ 資料夾（內含各類別子資料夾的影像）"
        f"或 cifar-10-batches-py/。目前該資料夾內容：{tree}"
    )


# --------------------------------------------------------------------------- #
# Query / Gallery 切分
# --------------------------------------------------------------------------- #
def balanced_sample(targets: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """各類別盡量平衡地抽樣 n 筆索引。"""
    n_cls = int(targets.max()) + 1
    per = n // n_cls
    idx: List[int] = []
    for c in range(n_cls):
        pool = np.where(targets == c)[0]
        idx.extend(rng.choice(pool, min(per, len(pool)), replace=False).tolist())
    rest = n - len(idx)
    if rest > 0:  # n 無法被類別數整除時，從剩餘樣本補足
        others = np.setdiff1d(np.arange(len(targets)), np.array(idx, dtype=int))
        idx.extend(rng.choice(others, min(rest, len(others)), replace=False).tolist())
    return np.array(idx)


def build_query_gallery(train_src, test_src, num_query: int, gallery_size: Optional[int], seed: int):
    """回傳 (query_source, query_idx, gallery_source, gallery_idx)。

    有 train 與 test：Query 從 test 平衡抽樣，Gallery 取 train（可再隨機縮小到 gallery_size）。
    只有 test：Query 與 Gallery 都來自 test，並確保兩者互不重疊。
    """
    rng = np.random.RandomState(seed)
    query_idx = balanced_sample(test_src.targets, num_query, rng)
    if train_src is not None:
        gallery_src, pool = train_src, np.arange(len(train_src))
    else:
        logger.warning("找不到 train，Gallery 改由 test 扣除 Query 後的剩餘樣本組成。")
        gallery_src, pool = test_src, np.setdiff1d(np.arange(len(test_src)), query_idx)
    if gallery_size and gallery_size < len(pool):
        pool = np.sort(rng.choice(pool, gallery_size, replace=False))
    return test_src, query_idx, gallery_src, pool


class SourceDataset(Dataset):
    """從 Source 取指定索引的影像，轉成骨幹輸入張量（放大到 224x224）。"""

    def __init__(self, source, indices: np.ndarray, transform):
        self.source, self.indices, self.transform = source, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.transform(Image.fromarray(self.source.get_image(int(self.indices[i]))))


@torch.no_grad()
def extract_features(model, loader: DataLoader, device: torch.device, desc: str) -> torch.Tensor:
    """抽取整個 DataLoader 的特徵向量（CPU float32）。"""
    model.eval()
    feats = []
    use_amp = device.type == "cuda"
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch)
        feats.append(out.float().cpu())
    gc.collect()  # 在主行程內先回收 DataLoader 的 worker，避免之後被子行程回收而印出 Exception ignored
    return torch.cat(feats, dim=0)


def run(
    data_dir: Path,
    result_dir: Path,
    device: torch.device,
    seed: int = 42,
    num_query: int = 100,
    gallery_size: Optional[int] = None,
    batch_size: int = 128,
    num_workers: int = 2,
    topk: int = 5,
    num_vis: int = 5,
    allow_download: bool = False,
) -> pd.DataFrame:
    """執行 Quiz 1，回傳對比表格 DataFrame。"""
    set_seed(seed)
    result_dir = ensure_dir(result_dir)

    # 1) 載入本地資料並切分
    train_src, test_src, class_names = load_local_cifar10(Path(data_dir), allow_download)
    q_src, query_idx, g_src, gallery_idx = build_query_gallery(
        train_src, test_src, num_query, gallery_size, seed
    )
    logger.info("類別：%s", class_names)
    logger.info("Query=%d 張（來自 test），Gallery=%d 張（來自 %s）",
                len(query_idx), len(gallery_idx), "train" if train_src is not None else "test")
    transform = build_imagenet_transform(224)
    pin = device.type == "cuda"

    def make_loader(src, idx) -> DataLoader:
        return DataLoader(
            SourceDataset(src, idx, transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        )

    q_loader, g_loader = make_loader(q_src, query_idx), make_loader(g_src, gallery_idx)
    q_labels = torch.from_numpy(q_src.targets[query_idx]).long()
    g_labels = torch.from_numpy(g_src.targets[gallery_idx]).long()

    # 2) 逐一模型：抽特徵 → 檢索 → 評估
    records, vis_rows = [], []
    vis_pos = np.linspace(0, len(query_idx) - 1, num=min(num_vis, len(query_idx)), dtype=int)
    for display_name, key in BACKBONES.items():
        logger.info("===== %s =====", display_name)
        model, feat_dim = build_image_backbone(key)
        model = model.to(device)

        t0 = time.time()
        q_feat = extract_features(model, q_loader, device, f"{key} query")
        g_feat = extract_features(model, g_loader, device, f"{key} gallery")
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        # L2 Normalization 後，內積即為 Cosine Similarity
        q_feat, g_feat = F.normalize(q_feat, dim=1), F.normalize(g_feat, dim=1)
        sim = q_feat @ g_feat.T  # [Q, G]
        metrics, top_idx, top_scores = topk_retrieval_metrics(sim, q_labels, g_labels, k=topk)

        records.append(
            {
                "Model": display_name,
                "Feature Dim": feat_dim,
                "Params (M)": round(count_parameters(model) / 1e6, 2),
                "Query / Gallery": f"{len(query_idx)} / {len(gallery_idx)}",
                "Top-1 Acc (%)": round(metrics["top1"], 2),
                f"Top-{topk} Acc (%)": round(metrics[f"top{topk}"], 2),
                f"Precision@{topk} (%)": round(metrics[f"precision@{topk}"], 2),
                "Extract Time (s)": round(elapsed, 1),
            }
        )

        for p in vis_pos:
            vis_rows.append(
                {
                    "model": display_name,
                    "query_img": q_src.get_image(int(query_idx[p])),
                    "query_label": int(q_labels[p]),
                    "retrieved": [
                        (g_src.get_image(int(gallery_idx[j])), int(g_labels[j]), float(s))
                        for j, s in zip(top_idx[p].tolist(), top_scores[p].tolist())
                    ],
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 3) 輸出
    plot_retrieval_samples(vis_rows, class_names, result_dir / "retrieval_samples.png")
    df = pd.DataFrame(records)
    save_table(df, result_dir / "metrics.csv")
    return df