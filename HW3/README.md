# 影像分類作業 (CNN Image Classification)

本專案實作了從資料集準備、Plain CNN 自行設計、經典 Backbone (ResNet18) 轉移學習，到資料擴增、Kernel 視覺化與 XAI 模型解釋之完整流程。

---
## 作業說明
* **Quiz 1：準備資料集**
  * 選定大型影像資料集並說明其資料來源、樣本內容、類別數與影像維度
  * 制定嚴謹的資料劃分策略（包含建立訓練集、驗證集與測試集），記錄於 `split.csv`

* **Quiz 2：訓練 CNN 影像分類模型**
  * **Plain CNN**：完成模型訓練，並呈現 Top-1 / Top-5 Accuracy，使用 Plain CNN 預測 Testing Dataset，並整理預測結果
  * **經典 CNN Backbone（遷移學習）**：完成模型訓練，並呈現 Top-1 / Top-5 Accuracy ，使用經典 CNN 模型預測 Testing Dataset，並整理預測結果
  * **效能與複雜度評估**：繪製各類別的 ROC Curve，計算  Macro-AUC ，並比較 Accuracy 表現與參數量大小
  * **超參數實驗**：探討不同超參數設定對模型收斂與準確率之影響

* **Quiz 3：提升泛化能力**
  * **資料增強**：為 Plain CNN 與 ResNet18 加入Data Augmentation，驗證改善程度
  * **卷積核特徵視覺化**：可視化第一層卷積核，並深入分析其中兩組 Kernel之特徵擷取功能
  * **可解釋性 AI (XAI)**：利用 Grad-CAM 產生物件類別激活熱力圖，驗證模型預測時所關注之關鍵影像區域
    
## 專案結構

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
```

## 執行環境
### 依賴套件清單
* Python >= 3.10
* torch == 2.5.1+cu121
* torchvision == 0.20.1+cu121
* numpy
* pandas
* Pillow
* opencv-python
* matplotlib
* scikit-learn
* tqdm

## 執行方式
開啟 Jupyter Notebook 循序執行 main.ipynb



