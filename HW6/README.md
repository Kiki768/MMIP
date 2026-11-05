# MMIP HW6：LLM / VLM 評測與 DPO 微調

所有實驗結果、表格與圖都在 **[main.ipynb](main.ipynb)** 裡，依序分成三個章節：

| 章節 | 內容 |
|---|---|
| 作業一 | Qwen2.5-7B-Instruct × 3 種 Prompt 策略（zero-shot / few-shot / CoT）× 3 種精度（FP16 / INT8 / INT4 NF4），VRAM 量測與 LLM-as-a-Judge 評分 |
| 作業 B | Qwen2-VL-7B-Instruct vs LLaVA-v1.6-Mistral-7B，20 題視覺測試（分類、定位、密集描述、圖表/OCR），客觀指標＋評審評分＋成對比較 |
| 作業 C | Qwen2.5-1.5B-Instruct + LoRA + DPO / RPO 微調「嚴格繁體中文輸出」；進階：COCO 幻覺抑制資料集、Qwen2-VL-7B 的 LoRA + DPO 微調、影像擾動魯棒性測試 |

## 資料夾結構

```
MMIPHW6/
├── main.ipynb              主程式：呼叫 src/ 的腳本並呈現所有結果
├── README.md
├── requirements.txt
├── src/                    實作程式碼
│   ├── common/
│   │   ├── paths.py            專案路徑
│   │   ├── gpu_monitor.py      VRAM 監測（torch allocator + NVML 背景取樣）
│   │   ├── judge.py            LLM-as-a-Judge 客戶端（OpenAI 相容 / Anthropic）
│   │   └── notebook_utils.py   notebook 用的小工具
│   ├── hw1/
│   │   ├── run_llm_eval.py     載入 → 量化 → 批次推論 → VRAM 量測 → 評分
│   │   └── plot_results.py     作業一圖表
│   ├── hw2/
│   │   ├── make_synthetic_images.py   下載 COCO 圖片、產生合成圖表/OCR 圖
│   │   ├── run_vlm.py                 兩個 VLM 的推論
│   │   └── judge_vlm.py               客觀指標 + pointwise / pairwise 評分
│   └── hw3/
│       ├── build_datasets.py   合成 DPO 成對資料、測試集、MMLU 抽樣
│       ├── train_dpo.py        TRL DPOTrainer + LoRA
│       ├── evaluate.py         微調前後比較（繁中指標、MMLU、McNemar 檢定）
│       ├── zh_metrics.py       簡體字判定（OpenCC）
│       ├── build_halluc_data.py   POPE 風格幻覺評測集與 VLM-DPO 成對資料
│       ├── robustness_eval.py     影像擾動魯棒性測試
│       ├── train_vlm_dpo.py       Qwen2-VL 的 LoRA + DPO 微調（可中斷續跑）
│       ├── eval_vlm_hallucination.py  微調前後描述圖片的幻覺比較
│       └── run_vlm_dpo.sh         VLM 微調的完整流程（訓練 → 評測 → 重跑 notebook）
└── data/                   輸入資料與實驗輸出
    ├── fonts/              Noto Sans CJK TC（圖表中文字型）
    ├── hw1/  benchmark.json、results/
    ├── hw2/  vlm_benchmark.json、images/、results/
    └── hw3/  dpo_train.jsonl、target_test.jsonl、general_bench.jsonl、
              coco/（標註、POPE 題目、幻覺 DPO 資料）、adapters/（LoRA 權重）、results/
```

## 環境

實測環境：V100 32GB、Python 3.12、CUDA 12.1。

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

需要一組 API key，在啟動 Jupyter 之前於終端機設定。三種都可以：

```bash
# Google Gemini（OpenAI 相容介面，有免費額度）
export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export OPENAI_API_KEY=<Gemini key>
export JUDGE_MODEL=gemini-3.5-flash-lite   # 免費額度較多；gemini-2.5-flash 已停用，flash 免費版每天只有 20 次

# OpenAI（預設 gpt-4o-mini）
export OPENAI_API_KEY=sk-...

# Anthropic
export JUDGE_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... JUDGE_MODEL=claude-haiku-5-5
```

## 執行

用 Jupyter / VS Code 開啟 `main.ipynb`，選擇 `.venv` 的 Python kernel，從頭依序執行即可；或在終端機一次執行完：

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 main.ipynb
```

每個耗時步驟都會先檢查 `data/` 裡是否已有結果，有就直接讀取，沒有才執行；把 notebook 開頭的 `FORCE_RERUN` 設成 `True` 可全部重跑（V100 約 5~6 小時，其中 VLM 微調流程約 3 小時）。沒有設定 API key 時，評分步驟會自動略過，其他部分照常執行。

各腳本也可以單獨執行，參數用 `python src/<hw>/<script>.py --help` 查看。
