# HW5：Multimodal Deep Learning

## 作業說明

| 任務 | 內容 | 主要模型 |
|---|---|---|
| Quiz 1 | 影像檢索（CIFAR-10） | ResNet-50 vs ViT-B/16 |
| Quiz 2 | 多模態分類（Hateful Memes） | (ResNet-50 / ViT-B/16) + BERT |
| Quiz 3 | 影像描述生成（Flickr8k） | CLIP + MLP + GPT-2 |

## 專案結構

```
HW5/
├── dataset/
│   ├── quiz1/                 # CIFAR-10：train/、test/（各類別一個資料夾）
│   ├── quiz2/                 # Hateful Memes：img/、train.jsonl、dev.jsonl、test.jsonl
│   └── quiz3/                 # Flickr8k：Flickr8k_Dataset/、Flickr8k_text/
├── src/
│   ├── utils.py               # 共用工具：骨幹模型、指標、畫圖、表格輸出
│   ├── quiz1_retrieval.py     # Quiz 1：影像檢索
│   ├── quiz2_multimodal.py    # Quiz 2：多模態分類
│   └── quiz3_captioning.py    # Quiz 3：CLIP Image Captioning
├── result/
│   ├── quiz1/                 # retrieval_samples.png、metrics.csv
│   ├── quiz2/                 # metrics.csv、training_curves.png、prediction_samples_*.png
│   └── quiz3/                 # caption_samples.png、metrics.csv、generated_captions.csv
├── main.ipynb                 
└── README.md
```

## 執行環境

- Python 3.12
- 主要套件：`torch`、`torchvision`、`transformers`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`、`nltk`、`bert-score`、`tqdm`、`Pillow`、`ipywidgets`

```bash
pip install torch torchvision transformers numpy pandas matplotlib \
            scikit-learn nltk bert-score tqdm pillow ipywidgets
```

## 執行方式

依序執行 main.ipynb 中各 Quiz 的儲存格
