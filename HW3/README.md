# 影像分類作業 (CNN Image Classification)

本專案實作了從資料集準備、Plain CNN 自行設計、經典 Backbone (ResNet18) 轉移學習，到資料擴增、Kernel 視覺化與 XAI 模型解釋之完整流程。

---
## 作業說明
* **Quiz 1：準備資料集 (Dataset Preparation)**[cite: 4]
  * 選定大型影像資料集並說明其資料來源、樣本內容、類別數與影像維度[cite: 4]。
  * 制定嚴謹的資料劃分策略（包含建立訓練集、驗證集與測試集，如 K-Fold / 分層抽樣），記錄於 `split.csv`[cite: 3, 4]。
  * 明確定義影像分類之任務目標與輸入／輸出規格[cite: 4]。

* **Quiz 2：訓練 CNN 影像分類模型 (CNN Classification Models)**[cite: 5]
  * **Plain CNN**：自定義多層卷積神經網路架構，進行模型訓練並計算 Top-1 與 Top-5 Accuracy[cite: 5]。
  * **經典 CNN Backbone（遷移學習）**：採用經典骨幹網路（如 ResNet18）進行 Transfer Learning[cite: 3, 5]，對比其分類效能。
  * **效能與複雜度評估**：對比兩模型之參數量（Parameters）、計算量（FLOPs）、繪製各類別 ROC 曲線並評估 Macro-AUC[cite: 5]。
  * **測試集預測與輸出**：將測試集預測結果匯出至 `plain_test_pred.csv` 與 `resnet18_test_pred.csv`[cite: 3, 5]。
  * **超參數實驗**：探討不同學習率（Learning Rate）與優化器（Optimizer）對模型收斂與準確率之影響[cite: 3, 5]。

* **Quiz 3：提升泛化能力與可解釋性 (Generalization & Explainability)**[cite: 6]
  * **資料增強 (Data Augmentation)**：為 Plain CNN 與 ResNet18 加入隨機裁切、水平翻轉、色彩抖動等擴增策略，驗證過擬合改善程度[cite: 6]。
  * **卷積核特徵視覺化 (Kernel Visualization)**：可視化第一層卷積核（`kernels_all.png`），並深入分析其中兩組 Kernel（`kernels_two.png`）之特徵擷取功能（如邊緣偵測、色彩對比）[cite: 3, 6]。
  * **可解釋性 AI (XAI)**：利用 Grad-CAM 產生物件類別激活熱力圖，驗證模型預測時所關注之關鍵影像區域[cite: 6]。
## 專案結構 (Project Structure)

```text
HW3/
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

