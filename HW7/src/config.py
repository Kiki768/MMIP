"""全域設定：API 金鑰來源、模型參數、路由部門定義、成本表與路徑。

設計原則
--------
* 不在程式碼中寫死任何金鑰。Claude 後端由官方 SDK 依序解析
  ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` / ``ant auth login`` profile。
* ``LLM_BACKEND`` 決定 LLM 類實驗（Exp A / C、影像感知、對話回覆）所使用的後端：
    - ``"auto"``   ：有 Claude 憑證 → Claude；否則有 CUDA 且本機模型已下載 → local；否則 offline
    - ``"claude"`` ：強制使用 Claude API（付費）
    - ``"local"``  ：強制使用本機開源 VLM（Qwen2-VL-2B-Instruct，免費，需 GPU）
    - ``"offline"``：強制使用可重現的離線模擬後端（無需網路、結果可重現）
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路徑
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

HW1_ARCH_PNG = RESULT_DIR / "hw1_architecture.png"
HW1_SPEC_JSON = RESULT_DIR / "hw1_spec_example.json"
HW2_BENCHMARK_JSON = RESULT_DIR / "hw2_benchmark.json"
HW2_COMPARISON_CSV = RESULT_DIR / "hw2_comparison.csv"
HW2_COMPARISON_PNG = RESULT_DIR / "hw2_comparison.png"
HW3_LOG_JSON = RESULT_DIR / "hw3_ux_security_logs.json"

# ---------------------------------------------------------------------------
# LLM 後端與模型參數
# ---------------------------------------------------------------------------
LLM_BACKEND = os.getenv("CHATBOT_LLM_BACKEND", "auto")  # auto | claude | local | offline

# 本機開源 VLM（免費、免金鑰；需 NVIDIA GPU，首次執行自動從 Hugging Face 下載約 4.5GB）
LOCAL_VLM_MODEL = os.getenv("CHATBOT_LOCAL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
LOCAL_MAX_NEW_TOKENS = 640
LOCAL_IMAGE_MAX_PIXELS = 640 * 28 * 28   # 限制影像 token 數（約 640 個），避免 6GB 顯存不足
LOCAL_IMAGE_MIN_PIXELS = 64 * 28 * 28
# Exp B：對本機小型 LLM 做 LoRA SFT（PEFT），讓模型「自己學會」輸出 RoutingDecision JSON
SFT_BASE_MODEL = os.getenv("CHATBOT_SFT_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
SFT_LORA_R = 16
SFT_LORA_ALPHA = 32
SFT_EPOCHS = 2
SFT_LR = 2e-4
SFT_BATCH = 8            # 有效 batch（micro-batch 2 × 梯度累積 4，峰值顯存約 4GB）
SFT_MAX_LEN = 512
# 本機推論的等效成本 = 推論秒數 × GPU 功耗 × 電價（筆電 RTX 3050 約 60W；台灣電價約 NT$4/kWh ≈ US$0.12）
LOCAL_GPU_WATTS = 60
ELECTRICITY_USD_PER_KWH = 0.12

CLAUDE_MODEL = os.getenv("CHATBOT_CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CHATBOT_CLAUDE_EFFORT", "low")  # 分流屬於分類任務，low 即可
CLAUDE_MAX_TOKENS = 4096
CLAUDE_TIMEOUT_S = 60.0

# Claude 計價（USD / 1M tokens）— 取自 Anthropic 官方價目（2026-11）
PRICING_USD_PER_MTOK = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
}
# 本地模型（Exp B / D）沒有 API 費用；以「CPU 推論電費 + 機器攤提」粗估之等效成本
LOCAL_COST_USD_PER_1K_REQ = {"lora_sft": 0.002, "system_one": 0.0001}

# 離線模擬後端的延遲模型（模擬雲端 LLM：網路 RTT + 首 token 時間 + 逐 token 生成）
OFFLINE_LATENCY_MODEL = {
    "base_s": 0.45,            # RTT + time-to-first-token
    "per_output_token_s": 0.012,
    "per_input_token_s": 0.00004,
}
# 離線模擬「純 Prompt 輸出 JSON」時的格式錯誤機率（依文獻與實務經驗設定的近似值）
OFFLINE_FREEFORM_FORMAT_ERROR_RATE = 0.22

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 路由部門設定
# ---------------------------------------------------------------------------
DEPARTMENTS: dict[str, dict] = {
    "billing": {
        "name_zh": "帳務與付款",
        "description": "扣款、重複扣款、發票、退款進度查詢、付款失敗",
        "required_slots": ["order_id"],
        "keywords": ["扣款", "付款", "刷卡", "發票", "帳單", "金額", "重複", "信用卡", "收據", "費用", "invoice", "charge", "billing"],
    },
    "technical_support": {
        "name_zh": "技術支援",
        "description": "App/網站錯誤、登入以外的系統故障、裝置連線、錯誤代碼",
        "required_slots": ["error_code"],
        "keywords": ["錯誤", "當機", "閃退", "無法開啟", "error", "bug", "連不上", "藍牙", "韌體", "更新失敗", "畫面", "代碼", "app"],
    },
    "logistics": {
        "name_zh": "物流配送",
        "description": "出貨進度、配送延遲、包裹遺失、改地址",
        "required_slots": ["order_id"],
        "keywords": ["物流", "配送", "包裹", "出貨", "到貨", "宅配", "運送", "追蹤", "還沒收到", "寄送", "地址", "超商取貨"],
    },
    "returns_refunds": {
        "name_zh": "退換貨",
        "description": "商品瑕疵、破損、退貨、換貨、鑑賞期",
        "required_slots": ["order_id", "product"],
        "keywords": ["退貨", "換貨", "破損", "瑕疵", "壞掉", "裂", "退款", "不合", "鑑賞期", "刮傷", "損壞"],
    },
    "account_security": {
        "name_zh": "帳號安全",
        "description": "盜用、異常登入、密碼重設、雙因素驗證",
        "required_slots": [],
        "keywords": ["盜用", "被盜", "密碼", "登入", "異常", "驗證碼", "帳號", "2fa", "鎖住", "釣魚"],
    },
    "sales": {
        "name_zh": "售前諮詢",
        "description": "商品規格、價格、優惠、庫存、方案比較",
        "required_slots": [],
        "keywords": ["規格", "價格", "優惠", "折扣", "推薦", "比較", "庫存", "方案", "多少錢", "有沒有", "型號", "購買"],
    },
    "human_agent": {
        "name_zh": "真人客服",
        "description": "使用者明確要求真人、高度不滿、或機器無法處理之情境",
        "required_slots": [],
        "keywords": ["真人", "客服人員", "主管", "投訴", "申訴", "人工"],
    },
}
DEPARTMENT_IDS = list(DEPARTMENTS.keys())

# 商品目錄（實體抽取用）
PRODUCT_CATALOG = [
    "SoundMax 藍牙耳機 X2", "SoundMax 藍牙喇叭 S1", "AirFlow 空氣清淨機 P3",
    "SmartWatch Pro 7", "PowerBank 20000", "HomeCam 監視器 C2", "SoundMax 耳罩式耳機 H5",
]

# System One 信心門檻：低於此值即升級至 System Two（LLM + 受限解碼）
SYSTEM_ONE_CONFIDENCE_THRESHOLD = 0.55

# Guardrail 設定
MAX_USER_MESSAGE_CHARS = 2000
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}
