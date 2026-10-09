"""Quiz 4：Captioning 的訓練、生成與端到端計時。

與 train.py（分類用）分開，因為 Captioning 的 loss 是逐字的 cross-entropy（teacher forcing），
每個 batch 是（影像特徵, 字 id 序列），不是（輸入, 標籤）。這個檔案不會動到 train.py。
"""
import time

import torch
import torch.nn as nn
from PIL import Image

from .caption_data import build_encoder, eval_transform
from .models.captioning import ids_to_caption


def _epoch(model, loader, criterion, device, optimizer=None, clip=None):
    train = optimizer is not None
    model.train(train)
    total, n_tok, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for feats, ids in loader:
            feats, ids = feats.to(device), ids.to(device)
            inp, tgt = ids[:, :-1], ids[:, 1:]  # 輸入 <start> w1 ... wn，預測 w1 ... wn <end>
            logits = model(feats, inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            mask = tgt != model.pad_id
            n = int(mask.sum())
            total += loss.item() * n
            n_tok += n
            correct += int(((logits.argmax(-1) == tgt) & mask).sum())
    return total / n_tok, correct / n_tok


def train_caption(model, train_loader, val_loader, device, epochs=15, lr=5e-4, clip=5.0,
                  patience=None, ckpt_path=None, name="caption"):
    """teacher forcing 訓練；以驗證集 loss 挑選最佳 epoch，結束時還原成最佳權重。

    history：train_loss / val_loss（每字的 cross-entropy）、train_acc / val_acc（逐字準確率）、
    val_ppl（驗證集 perplexity，越低越好）、epoch_time、total_time。
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id)  # <pad> 位置不計入 loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_ppl", "epoch_time")}
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _epoch(model, train_loader, criterion, device, optimizer, clip)
        va_loss, va_acc = _epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        dt = time.time() - t0
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, float(torch.exp(torch.tensor(va_loss))), dt)):
            hist[k].append(v)
        if va_loss < best:
            best, best_epoch, bad = va_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[{name}] epoch {epoch:02d}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} ppl {hist['val_ppl'][-1]:.1f} acc {va_acc:.3f} | {dt:.1f}s")
        if patience and bad >= patience:
            print(f"[{name}] 驗證 loss 連續 {patience} 個 epoch 沒有進步，提前停止")
            break

    hist["total_time"] = time.time() - t_start
    hist["best_epoch"], hist["best_val_loss"] = best_epoch, best
    model.load_state_dict(best_state)
    if ckpt_path is not None:
        torch.save(best_state, ckpt_path)
    return hist


@torch.no_grad()
def generate_captions(model, feats, vocab, device, method="greedy", beam_size=3, max_len=20, batch_size=256):
    """對一批已抽好的特徵生成描述（用來評估整個測試集）。method: 'greedy' 或 'beam'。"""
    model.to(device).eval()
    caps = []
    if method == "greedy":
        for i in range(0, len(feats), batch_size):
            ids = model.greedy(feats[i : i + batch_size].to(device), max_len)
            caps += [ids_to_caption(r.tolist(), vocab) for r in ids]
    elif method == "beam":
        for f in feats:
            caps.append(ids_to_caption(model.beam_search(f.to(device), beam_size, max_len), vocab))
    else:
        raise ValueError("method 需為 'greedy' 或 'beam'")
    return caps


class CaptionPipeline:
    """端到端：圖片檔 -> 前處理 -> 編碼器 -> 解碼器 -> 句子，並量測每張圖的耗時。

    量到的時間包含讀檔與前處理，才能和「把圖片送去 Gemini 並等回覆」的耗時公平比較。
    """

    def __init__(self, encoder_name, decoder, vocab, device, pretrained=True):
        self.device, self.vocab, self.decoder = device, vocab, decoder.to(device).eval()
        self.encoder = build_encoder(encoder_name, pretrained).to(device).eval()
        self.transform = eval_transform()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def caption(self, image, method="greedy", beam_size=3, max_len=20):
        """image 可以是路徑或 PIL.Image。回傳 (描述, 耗時秒數)。"""
        self._sync()
        t0 = time.perf_counter()
        img = Image.open(image) if not isinstance(image, Image.Image) else image
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.encoder(x).float()
        if method == "greedy":
            ids = self.decoder.greedy(feat, max_len)[0].tolist()
        else:
            ids = self.decoder.beam_search(feat, beam_size, max_len)
        text = ids_to_caption(ids, self.vocab)
        self._sync()
        return text, time.perf_counter() - t0


def time_pipeline(pipeline, paths, method="greedy", beam_size=3, max_len=20, warmup=3):
    """逐張（batch size = 1）生成描述並計時；先暖機幾張避免第一次呼叫的初始化時間混進去。
    回傳 (描述 list, 秒數 list)。"""
    for p in paths[:warmup]:
        pipeline.caption(p, method, beam_size, max_len)
    caps, secs = [], []
    for p in paths:
        c, s = pipeline.caption(p, method, beam_size, max_len)
        caps.append(c)
        secs.append(s)
    return caps, secs
