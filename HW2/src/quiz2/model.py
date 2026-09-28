"""Step 3：建立 Multi-Layer Perceptron"""
import torch.nn as nn

# in_dim：輸入特徵數量；hidden：隱藏層神經元數量；dropout：隨機失活比例
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64, 32), dropout=0.0):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # 輸出 1 個 logit（二元分類）
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape: (batch,)