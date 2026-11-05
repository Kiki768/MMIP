評審模型：gemini-3.5-flash-lite

## 總體指標

|                       |   Qwen2-VL-7B-Instruct |   llava-v1.6-mistral-7b-hf |
|:----------------------|-----------------------:|---------------------------:|
| overall               |                  8.49  |                      5.305 |
| accuracy              |                  7.95  |                      4.35  |
| completeness          |                  8.1   |                      5.3   |
| hallucination         |                  9.15  |                      5.7   |
| instruction_following |                  9.35  |                      7.2   |
| keyword_recall        |                  0.688 |                      0.551 |
| bbox_mIoU             |                  0.782 |                      0.651 |
| halluc_trap_pass      |                  0     |                      1     |
| new_tokens            |                 96.35  |                    208.35  |
| latency_s             |                  4.318 |                     13.876 |
| pairwise_win_rate     |                  0.8   |                      0     |
| pairwise_tie_rate     |                  0.2   |                      0.2   |
| peak_vram_gib         |                 17.57  |                     22.94  |

## 各類別 overall（1–10）

| category      |   Qwen2-VL-7B-Instruct |   llava-v1.6-mistral-7b-hf |
|:--------------|-----------------------:|---------------------------:|
| 圖表/OCR      |                   9.22 |                       3.04 |
| 密集細節描述  |                   7.9  |                       4.21 |
| 影像分類      |                   9.84 |                       7.1  |
| 物件位置/偵測 |                   6.74 |                       7.1  |

## Pairwise 逐題勝負

| id     | category      | order1                            | order2                            | winner                    |
|:-------|:--------------|:----------------------------------|:----------------------------------|:--------------------------|
| CLS-01 | 影像分類      | Qwen/Qwen2-VL-7B-Instruct         | llava-hf/llava-v1.6-mistral-7b-hf | tie                       |
| CLS-02 | 影像分類      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| CLS-03 | 影像分類      | Qwen/Qwen2-VL-7B-Instruct         | llava-hf/llava-v1.6-mistral-7b-hf | tie                       |
| CLS-04 | 影像分類      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| CLS-05 | 影像分類      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| LOC-01 | 物件位置/偵測 | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| LOC-02 | 物件位置/偵測 | llava-hf/llava-v1.6-mistral-7b-hf | Qwen/Qwen2-VL-7B-Instruct         | tie                       |
| LOC-03 | 物件位置/偵測 | Qwen/Qwen2-VL-7B-Instruct         | llava-hf/llava-v1.6-mistral-7b-hf | tie                       |
| LOC-04 | 物件位置/偵測 | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| LOC-05 | 物件位置/偵測 | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| DEN-01 | 密集細節描述  | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| DEN-02 | 密集細節描述  | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| DEN-03 | 密集細節描述  | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| DEN-04 | 密集細節描述  | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| OCR-01 | 圖表/OCR      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| OCR-02 | 圖表/OCR      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| OCR-03 | 圖表/OCR      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| OCR-04 | 圖表/OCR      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| OCR-05 | 圖表/OCR      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
| OCR-06 | 圖表/OCR      | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct         | Qwen/Qwen2-VL-7B-Instruct |
