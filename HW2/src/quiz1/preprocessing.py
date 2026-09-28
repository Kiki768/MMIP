import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    zero_as_nan_cols = ['glucose_concentration', 'diastolic_blood_pressure',
                         'triceps_sf_thickness', 'serum_insulin', 'bmi']
    df[zero_as_nan_cols] = df[zero_as_nan_cols].replace(0, np.nan) 
    for col in zero_as_nan_cols:
        df[col] = df[col].fillna(df[col].median()) # 將缺失值以中位數填補
    return df

def split_and_scale(df, target_col='diabetes'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_val, y_train, y_val = train_test_split( # X_train 訓練集特徵, X_val 驗證集特徵, y_train 標籤(0或1), y_val 標籤(0或1)
        X, y, test_size=0.2, random_state=42, stratify=y) # 加了 stratify=y 以確保訓練集和驗證集的類別分佈相似
    scaler = StandardScaler() # 特徵標準化
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, y_train, y_val