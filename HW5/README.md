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
├── main.ipynb                 # 統一入口
└── README.md
```

## 執行環境

- Python 3.12
- 主要套件：`torch`、`torchvision`、`transformers`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`、`nltk`、`bert-score`、`tqdm`、`Pillow`、`ipywidgets`

```bash
pip install torch torchvision transformers numpy pandas matplotlib \
            scikit-learn nltk bert-score tqdm pillow ipywidgets
```

首次執行會自動下載預訓練模型（ResNet-50、ViT-B/16、`bert-base-uncased`、`openai/clip-vit-base-patch32`、`gpt2`，以及 BERTScore 使用的 `roberta-large`），需要網路連線。建議使用 GPU 執行。

## 執行方式

1. 以 Jupyter 或 VS Code 開啟 `main.ipynb`（notebook 需放在 `HW5/` 根目錄）。
2. 在「參數設定」cell 修改 `parse_args([...])`：

```python
args = parse_args([
    "--task", "all",          # all / quiz1 / quiz2 / quiz3
    "--device", "cuda",       # auto / cpu / cuda / mps
    # "--quick",              # 小資料冒煙測試，確認流程可跑通
])
```

3. 由上而下依序執行各 cell。某個 Quiz 失敗時只會印出原因，不影響其他 Quiz。

常用參數：

| 參數 | 說明 | 預設 |
|---|---|---|
| `--task` | 要執行的任務 | all |
| `--device` | 運算裝置 | auto |
| `--quiz1_num_query` | Quiz 1 的 Query 數量 | 100 |
| `--quiz1_gallery_size` | Quiz 1 的 Gallery 張數 | train 全部 |
| `--quiz2_combos` | Quiz 2 的模型組合（A / B / C） | A B |
| `--quiz2_epochs` | Quiz 2 訓練 epoch 數 | 5 |
| `--quiz2_finetune` | Quiz 2 解凍骨幹端到端微調 | 關閉 |
| `--quiz3_epochs` | Quiz 3 訓練 epoch 數 | 3 |
| `--quiz3_week4_baseline` | Week 4 baseline 指標 JSON 路徑 | `result/quiz3/week4_baseline.json` |
| `--num_workers` | DataLoader 執行緒數（Windows 出錯請設 0） | 2 |

執行結果會存放在 `result/quiz1/`、`result/quiz2/`、`result/quiz3/`。
