# 影像分類作業 (CNN Image Classification)

本專案實作了從資料集準備、Plain CNN 自行設計、經典 Backbone (ResNet18) 轉移學習，到資料擴增、Kernel 視覺化與 XAI 模型解釋之完整流程。

---

## 專案結構 (Project Structure)

```text
├── checkpoints/             # 存放各模型權重
├── images                   # 資料集
├── results                  # 訓練、評估後的結果
├── src/
│   ├── data.py              # 資料集讀取、前處理與 Augmentation Pipeline
│   ├── engine.py            # 訓練、驗證、測試與評估函式 (Top-1, Top-5, ROC/AUC)
│   └── models.py            # Plain CNN 架構定義與 ResNet18 模型封裝
├── AI.docx                  # AI相關輔助證明
├── main.ipynb               # 實驗主執行流程與視覺化分析
├── split.csv                # 資料集切分索引記錄
└── README.md

