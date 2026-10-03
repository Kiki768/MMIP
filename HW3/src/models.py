import torch.nn as nn
from torchvision import models

def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

class PlainCNN(nn.Module):
    """4 個卷積區塊 + Global Average Pooling + 全連接層"""
    def __init__(self, num_classes=37, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        chans = (3,) + tuple(widths)
        self.features = nn.Sequential(*[conv_block(chans[i], chans[i+1]) for i in range(len(widths))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(widths[-1], num_classes))
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def build_resnet18(num_classes=37, pretrained=True, freeze_backbone=False):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)  # 新的分類頭一定可訓練
    return m

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable