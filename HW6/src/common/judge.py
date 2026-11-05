"""
LLM-as-a-Judge 共用模組（作業 1、2 共用）。

支援兩種商用 API：
  - OpenAI   （預設，環境變數 OPENAI_API_KEY，模型 JUDGE_MODEL 預設 gpt-4o-mini）
  - Anthropic（JUDGE_PROVIDER=anthropic，環境變數 ANTHROPIC_API_KEY）

評審一律被要求輸出 JSON，回傳 dict；失敗會自動重試，最終失敗時回傳 {"error": ...}。
成功的評分會快取在 data/judge_cache/<模型>.jsonl：額度用完中斷後重跑，已評過的題目直接讀快取、不會重複呼叫。
"""
import base64
import hashlib
import json
import os
import re
import time

from .paths import DATA

PROVIDER = os.getenv("JUDGE_PROVIDER", "openai").lower()
DEFAULT_MODEL = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}[PROVIDER]
MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)

_client = None
_consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 5   # 連續失敗代表設定有誤（key、網址、模型名稱），直接中止，不要白跑完全部題目
CACHE_PATH = os.path.join(DATA, "judge_cache", re.sub(r"[^\w.-]", "_", MODEL) + ".jsonl")
_cache = None


def _cache_key(system_prompt, user_prompt, image_path):
    raw = json.dumps([MODEL, system_prompt, user_prompt, os.path.basename(image_path or "")], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    _cache[row["key"]] = row["result"]
    return _cache


def _get_client():
    global _client
    if _client is None:
        if PROVIDER == "openai":
            from openai import OpenAI
            _client = OpenAI()
        else:
            import anthropic
            _client = anthropic.Anthropic()
    return _client


def _image_part(path):
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    if PROVIDER == "openai":
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}}
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}


def _parse_json(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def judge(system_prompt, user_prompt, image_path=None, max_retries=5):
    """呼叫評審模型，回傳解析後的 JSON dict。temperature=0 以提高評分可重現性。"""
    global _consecutive_failures
    key = _cache_key(system_prompt, user_prompt, image_path)
    if key in _load_cache():
        return _cache[key]
    client = _get_client()
    content = [{"type": "text", "text": user_prompt}]
    if image_path:
        content.insert(0, _image_part(image_path))

    for attempt in range(max_retries):
        try:
            if PROVIDER == "openai":
                resp = client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": content}],
                )
                text = resp.choices[0].message.content
            else:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    temperature=0,
                    system=system_prompt + "\n只輸出一個 JSON 物件，不要有其他文字。",
                    messages=[{"role": "user", "content": content}],
                )
                text = resp.content[0].text
            result = _parse_json(text)
            _consecutive_failures = 0
            _cache[key] = result
            os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
            with open(CACHE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
            return result
        except Exception as e:  # 網路錯誤、rate limit、JSON 解析失敗都重試
            if attempt == max_retries - 1:
                _consecutive_failures += 1
                if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(f"評審連續 {_consecutive_failures} 次失敗，請檢查 API key、"
                                       f"OPENAI_BASE_URL 與 JUDGE_MODEL={MODEL}。最後的錯誤：{type(e).__name__}: {e}") from e
                return {"error": f"{type(e).__name__}: {e}"}
            time.sleep(min(60, 5 * 2 ** attempt))   # 免費方案常遇到 rate limit，等久一點再試
