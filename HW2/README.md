# MMIP HW2
## 作業說明
| Quiz | 主題 | 資料集 | 模型 |
|---|---|---|---|
| Quiz 1 | 機器學習分類任務 | Pima Indians Diabetes | Logistic Regression、Random Forest |
| Quiz 2 | 深度學習信用卡違約預測 | UCI Credit Card | MLP（PyTorch） |
| Quiz 3 | 模型表現評估（ROC / AUC） | UCI Credit Card | MLP、Random Forest |
## 專案結構
```text
HW2/
├── dataset/            # 輸入資料集 (quiz1 ~ quiz3)
├── result/             # 混淆矩陣、LOSS圖、ROC Curve等
├── src/                # 演算法核心模組
│   ├── quiz1.py
│   ├── quiz2.py
│   └── quiz3.py
├── AI.docx             # AI相關說明
├── main.ipynb          # 主程式    
└── README.md           # HW2 說明文件
```
## 執行環境
Python 版本：Python 3.11.9
## 執行方式
開啟 Jupyter Notebook 循序執行 main.ipynb
