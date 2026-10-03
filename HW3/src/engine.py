import os, time, torch, numpy as np, torch.nn as nn

def topk_correct(logits, y, ks=(1, 5)):
    _, pred = logits.topk(max(ks), dim=1)
    hit = pred.eq(y.unsqueeze(1))
    return [hit[:, :k].any(dim=1).sum().item() for k in ks]

@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    loss_sum, c1, c5, n = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
            if criterion: loss_sum += criterion(out, y).item() * len(y)
        a, b = topk_correct(out, y)
        c1 += a; c5 += b; n += len(y)
    return loss_sum / n, c1 / n, c5 / n

def fit(model, train_loader, val_loader, device, name, epochs=20, lr=1e-3,
        optimizer="adam", weight_decay=1e-4, label_smoothing=0.0):
    os.makedirs("checkpoints", exist_ok=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    hist = {k: [] for k in ["train_loss", "val_loss", "val_top1", "val_top5"]}
    best = 0.0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); n += len(y)
        sched.step()
        vl, v1, v5 = evaluate(model, val_loader, device, criterion)
        for k, v in zip(hist, [tl / n, vl, v1, v5]): hist[k].append(v)
        if v1 > best:
            best = v1; torch.save(model.state_dict(), f"checkpoints/{name}.pt")
        print(f"[{name}] ep{ep+1:02d}/{epochs} loss {tl/n:.3f} | val loss {vl:.3f} "
              f"top1 {v1:.3f} top5 {v5:.3f} | {time.time()-t0:.0f}s")
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt"))  # 載回最佳權重
    return hist

@torch.no_grad()
def predict(model, loader, device):
    """回傳 (N,37) 機率與真實標籤;順序與 loader 一致"""
    model.eval(); probs, ys = [], []
    for x, y in loader:
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x.to(device))
        probs.append(out.float().softmax(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)