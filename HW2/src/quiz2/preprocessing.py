"""Step 1~3：讀取資料、資料清理、Feature Scaling、切分 Train/Validation"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# HW2/src/quiz2/data.py -> parents[2] = HW2
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "dataset" / "UCI_Credit_Card.csv"
TARGET = "default.payment.next.month"

# ---- 欄位分組（依照資料集說明）----
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]                      # 類別：One-Hot
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]  # 還款狀態：有順序，當數值
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]                     # 帳單金額（可能為負）
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]                       # 繳款金額（>= 0）
MONEY = ["LIMIT_BAL"] + BILL_AMT + PAY_AMT                           # 金額欄位：先取 log

def clean_data(df):
    """Step 1：資料清理"""
    df = df.drop(columns=["ID"])  # ID 只是編號，對預測沒有意義

    # EDUCATION：說明中 5、6 都是 unknown，0 則完全沒定義 -> 統一歸為 5 (unknown)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 5, 6: 5})
    # MARRIAGE：0 沒定義 -> 歸為 3 (others)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    # PAY_x：說明只寫了 -1 與 1~9，但資料還有 -2（無消費）與 0（循環信用，有繳最低應繳）。
    # 三者都代表「沒有延遲」，而 1~8 代表延遲月數，數字大小有順序意義 -> 保留原值當數值特徵
    return df

def signed_log1p(x):
    """保留正負號的 log：sign(x) * log(1 + |x|)，處理 BILL_AMT 的負值（溢繳）"""
    # np.abs(x)先取絕對值，確保數值為正數，再log(1 + |x|)，x=0 時結果是 0，不會出錯
    # 最後乘上 np.sign(x) 保留正負號，x=0 時結果仍是 0
    # 100 萬 → 約 13.8；-5 萬 → 約 -10.8；0 → 0。原本相差百萬倍的數值，被壓縮到十幾的範圍內。
    return np.sign(x) * np.log1p(np.abs(x))

# 把原始 CSV 變成模型看得懂的數字
def load_data(val_size=0.2, seed=42):
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)

    # Step 2a：金額欄位高度右偏（少數人金額極大），先取 log 壓縮尺度
    df[MONEY] = signed_log1p(df[MONEY])

    # Step 2b：類別欄位 One-Hot
    df = pd.get_dummies(df, columns=CATEGORICAL, dtype=np.float32)

    y = df[TARGET].values.astype(np.float32)
    X_df = df.drop(columns=[TARGET])
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)

    # Step 3：切分，stratify=y 讓 train/val 違約比例一致（約 22%）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=seed
    )

    # Step 2c：StandardScaler 只套用在「數值欄位」，One-Hot 欄位維持 0/1
    #         且只用 training set fit，避免 data leakage
    num_idx = [i for i, c in enumerate(feature_names)
               if not c.startswith(tuple(f"{cat}_" for cat in CATEGORICAL))]
    scaler = StandardScaler()
    X_train_s, X_val_s = X_train.copy(), X_val.copy()
    X_train_s[:, num_idx] = scaler.fit_transform(X_train[:, num_idx])
    X_val_s[:, num_idx] = scaler.transform(X_val[:, num_idx])
    X_train_s, X_val_s = X_train_s.astype(np.float32), X_val_s.astype(np.float32)

    return X_train_s, X_val_s, y_train, y_val, feature_names

# 把資料切成一批一批送進模型
def make_loaders(X_train, X_val, y_train, y_val, batch_size=64):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) 
    # shuffle=True（只對 train）：每個 epoch 打亂資料順序，避免模型記住資料的排列方式，validation 不需要打亂。
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


