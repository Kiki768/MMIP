"""Step 4：訓練迴圈（支援 L2、Early Stopping、Learning Rate Scheduler）"""
import copy

import torch
import torch.nn as nn

# 跑一個epoch，training=True時會更新權重，training=False時只計算loss
def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb) # 計算loss
            if training:
                optimizer.zero_grad()
                loss.backward() # 反向傳播計算梯度
                optimizer.step() # 根據梯度更新權重
            total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3,
                optimizer_name="adam", weight_decay=0.0,
                early_stopping_patience=None, use_scheduler=False,
                device="cpu"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # ReduceLROnPlateau 的意思是「進入停滯期時降低學習率」
    # mode="min"：監控的數值越小越好（loss 本來就是越小越好）。
    # patience=5：val loss 連續 5 個 epoch 沒有下降，就算進入停滯。
    # factor=0.5：停滯時，學習率乘以 0.5。
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
                 if use_scheduler else None)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if early_stopping_patience and wait >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 有開 Early Stopping 時，還原到 val loss 最低的權重
    if early_stopping_patience and best_state is not None:
        model.load_state_dict(best_state)

    history["stopped_epoch"] = len(history["train_loss"])
    history["best_val_loss"] = best_val
    return history