"""分類任務通用的訓練 / 推論迴圈（Quiz 2 文字分類、Quiz 3 影像分類皆可使用）。

資料 loader 需回傳 (inputs, labels)。Captioning 的訓練方式不同，之後另寫 train_caption()。

Quiz 3 微調大型預訓練模型（ViT）時可開啟：
    amp=True     混合精度訓練（GPU 上更快、更省記憶體；CPU 上自動忽略）
    cosine=True  學習率以 cosine 曲線逐步下降（每個 batch 更新一次）
"""
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_epoch(model, loader, criterion, device, optimizer=None, clip=None, scaler=None, amp=False, scheduler=None):
    train = optimizer is not None
    model.train(train)
    use_amp = amp and device.type == "cuda"
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        desc = f"Epoch [{optimizer is not None and 'Train' or 'Val'}]"
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                step_executed = True  # 記錄 optimizer 是否有真正更新

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if clip:
                        scaler.unscale_(optimizer)  # 先還原梯度尺度，裁剪才有意義
                        nn.utils.clip_grad_norm_(model.parameters(), clip)
                    
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler.get_scale()
                    
                    # 若 scale_after < scale_before，代表梯度溢位，本次更新被跳過
                    if scale_after < scale_before:
                        step_executed = False
                else:
                    loss.backward()
                    if clip:
                        nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

                # 只有在 optimizer 真正更新成功時，才更新排程器
                if scheduler is not None and step_executed:
                    scheduler.step()
            bs = y.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            n += bs
    return total_loss / n, correct / n


def train_classifier(
    model,
    train_loader,
    val_loader,
    device,
    epochs=10,
    lr=1e-3,
    weight_decay=0.0,
    clip=1.0,
    ckpt_path=None,
    name="model",
    amp=False,
    cosine=False,
):
    """訓練並以驗證集準確率挑選最佳 epoch；結束時會把模型還原成最佳權重。

    只會更新 requires_grad=True 的參數（凍結的層不會被訓練）。
    回傳 history dict，包含每個 epoch 的 loss / acc / 耗時，以及總訓練時間。
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)  # weight_decay=0 時等同 Adam
    scheduler = None
    if cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")}
    best_acc, best_state, best_epoch = -1.0, None, 0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer, clip, scaler, amp, scheduler)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device, amp=amp)
        dt = time.time() - t0
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, dt)):
            history[k].append(v)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[{name}] epoch {epoch:02d}/{epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s"
        )

    history["total_time"] = time.time() - t_start
    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_acc
    model.load_state_dict(best_state)
    if ckpt_path is not None:
        torch.save(best_state, ckpt_path)
    return history


@torch.no_grad()
def predict(model, loader, device, amp=False):
    """對整個 loader 推論，回傳 (y_true, y_pred, probs)；probs 形狀為 (N, num_classes)。"""
    model.to(device).eval()
    use_amp = amp and device.type == "cuda"
    ys, probs = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        ys.append(y)
    probs = torch.cat(probs).numpy()
    y_true = torch.cat(ys).numpy()
    return y_true, probs.argmax(1), probs


def measure_inference_time(model, loader, device, amp=False):
    """量測對整個 loader 推論所需秒數（含資料搬移），供模型比較使用。"""
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    predict(model, loader, device, amp=amp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - t0
