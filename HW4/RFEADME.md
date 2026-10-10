# MMIP HW4

## 作業說明

| Quiz | 主題 | 內容 |
|---|---|---|
| 1 | 資料集準備 | IMDB(文字分類)、CIFAR-10(影像分類)、Flickr8k(Image Captioning) |
| 2 | 情感分析 | 訓練 RNN 並推論;訓練 LSTM 並與 RNN 比較 |
| 3 | Vision Transformer | 微調 ViT-B/16 分類;微調 ResNet50 並比較 Macro-AUC |
| 4 | Image Captioning | 訓練 captioning 模型;Gemini 判斷描述是否符合圖片;BLEU 量化;比較不同方法的表現與回應時間 |

## 專案結構

```
.
├── main.ipynb              # 主要執行入口
├── dataset/                # 資料集
│   ├── quiz2/              # IMDB 50K CSV
│   ├── quiz3/              # CIFAR-10(train/、test/)
│   └── quiz4/              # Flickr8k(Flickr8k_Dataset/、Flickr8k_text/)
├── result/                 # 模型訓練、測試圖表與表格
│   ├── quiz2/
│   ├── quiz3/
│   └── quiz4/
├── src/
│   ├── data_loader.py      # Quiz 1~3 資料載入與切分
│   ├── caption_data.py     # Quiz 4 資料與特徵抽取
│   ├── train.py            # 分類任務訓練 / 推論 / 計時
│   ├── train_caption.py    # Captioning 訓練、生成、計時
│   ├── eval_metrics.py     # 分類指標與繪圖
│   ├── eval_caption.py     # BLEU、Gemini 評估
│   └── models/
│       ├── base.py
│       ├── rnn.py
│       ├── lstm.py
│       ├── vision.py       # ViT、ResNet
│       └── captioning.py   # LSTM 解碼器
└── README.md
```

## 執行環境

- Python 3.10 以上，建議使用 GPU
- 安裝套件:

```bash
bert-score==0.3.13
google-genai==2.29.0
matplotlib==3.11.2
nltk==3.10.3
numpy==2.5.2
pandas==3.0.6
pillow==12.3.0
scikit-learn==1.9.1
torch==2.5.1+cu121
torchvision==0.20.1+cu121
tqdm==4.70.1
```

## 執行方式

依序執行 `main.ipynb` 中各 Quiz 的儲存格
