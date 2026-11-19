# 多模態多輪對話客服機器人

作業一～三：結構化分流規格 → 四種分流方法比較 → 多輪多模態客服機器人（含 UX 優化與安全防禦）。
報告與所有結果都在 `main.ipynb`。

## 資料夾結構

```
HW7/
├── main.ipynb            # 主報告：作業一～三的說明、程式與執行結果
├── requirements.txt      # 套件清單
├── README.md
├── src/
│   ├── config.py         # 設定（部門、模型名稱、門檻、路徑）
│   ├── schemas.py        # 結構化輸出格式 RoutingDecision（Pydantic）
│   ├── routing_engines.py  # 四種分流方法 A/B/C/D、串接、看圖模型
│   ├── evaluator.py      # Benchmark 資料與量化評估
│   ├── chatbot_core.py   # 多輪對話管理、UX、離題處理、安全防禦、測試情境
│   └── ui_app.py         # Gradio 聊天室介面
└── result/
    ├── hw1_architecture.png        # 系統架構圖
    ├── hw1_spec_example.json       # 輸出格式範例
    ├── hw2_benchmark.json          # 16 筆測試資料
    ├── hw2_comparison.csv / .png   # 四種方法比較結果
    ├── hw3_ux_security_logs.json   # 作業三測試情境紀錄
    └── hw3_ui_*.png                # UI 操作截圖
```

## 環境

* Windows 11、Python 3.11
* GPU：NVIDIA RTX 3050 Laptop（6GB），CUDA 12.1
* 主要套件：torch 2.5.1+cu121、transformers 4.57.6、peft 0.21、gradio 5.50

使用的模型（都是免費的開源模型，第一次執行時會自動從 Hugging Face 下載）：

| 用途 | 模型 | 大小 |
|---|---|---|
| 看圖、方法 A / C | `Qwen/Qwen2-VL-2B-Instruct` | 約 4.5 GB |
| 方法 B（LoRA 微調） | `Qwen/Qwen2.5-0.5B-Instruct` | 約 1 GB |

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 執行方式

**1. 報告（notebook）**

用 VS Code 或 Jupyter 打開 `main.ipynb`，執行全部儲存格（約 40 分鐘，含 B 的訓練）。
結果會同時更新到 `result/`。

**2. 聊天室 UI**

```bash
python -m src.ui_app
```

再用瀏覽器打開 http://127.0.0.1:7860 。

**執行模式**

程式會自動選擇後端，也可以用環境變數 `CHATBOT_LLM_BACKEND` 指定：

| 模式 | 說明 |
|---|---|
| `local` | 本機 GPU 跑開源模型（**本報告的數字用這個模式**） |
| `offline` | 不用 GPU、不用下載模型，用規則模擬，幾秒就跑完，但數字會跟報告不同 |
| `claude` | 使用 Claude API（需要 API key，會收費） |
| `auto`（預設） | 有 Claude 憑證 → claude；有 GPU 和模型 → local；都沒有 → offline |

例如沒有 GPU 時（PowerShell）：

```powershell
$env:CHATBOT_LLM_BACKEND="offline"
python -m src.ui_app
```