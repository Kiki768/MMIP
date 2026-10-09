"""Quiz 4 評估：BLEU、BERTScore、Gemini 判斷 / 生成（含耗時紀錄）與繪圖。

這是 Quiz 4 專用的新檔案，不會動到 eval_metrics.py。

Gemini 需要 API key，取得順序：函式參數 -> 環境變數 GEMINI_API_KEY / GOOGLE_API_KEY
-> Colab 的 Secrets（名稱 GEMINI_API_KEY）-> 執行時手動輸入。請不要把 key 寫進程式或 notebook。
"""
import json
import mimetypes
import os
import random
import re
import textwrap
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# 模型名稱可能隨時間更新；若出錯，用 list_gemini_models(client) 查目前可用的名稱後替換。
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


# ----------------------------------------------------------------------------
# BLEU / BERTScore
# ----------------------------------------------------------------------------
def _tok(s):
    return re.findall(r"[a-z0-9']+", str(s).lower())


def compute_bleu(hyps, refs):
    """corpus-level BLEU-1~4，每張圖用全部參考句（Flickr8k 為 5 句）。分數範圍 0~1。"""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError as e:
        raise ImportError("計算 BLEU 需要 nltk：pip install nltk") from e
    list_refs = [[_tok(r) for r in rs] for rs in refs]
    list_hyps = [_tok(h) for h in hyps]
    smooth = SmoothingFunction().method1
    weights = {
        "BLEU-1": (1, 0, 0, 0),
        "BLEU-2": (0.5, 0.5, 0, 0),
        "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25),
    }
    return {k: corpus_bleu(list_refs, list_hyps, weights=w, smoothing_function=smooth) for k, w in weights.items()}


def compute_bertscore(hyps, refs, model_type=None, lang="en", batch_size=64, device=None, rescale_with_baseline=False):
    """BERTScore（有多句參考時，對每張圖取與各參考句分數最高者）。

    model_type=None 時 lang='en' 會用 roberta-large（約 1.4GB，第一次執行要下載）；
    想快一點可傳 'distilbert-base-uncased'。
    """
    try:
        from bert_score import score
    except ImportError as e:
        raise ImportError("計算 BERTScore 需要 bert-score：pip install bert-score") from e
    P, R, F = score(
        list(hyps), [list(r) for r in refs], lang=lang, model_type=model_type, batch_size=batch_size,
        device=device, rescale_with_baseline=rescale_with_baseline, verbose=False,
    )
    return {"BERTScore-P": P.mean().item(), "BERTScore-R": R.mean().item(), "BERTScore-F1": F.mean().item()}


def evaluate_captions(hyps, refs, bertscore=True, bertscore_model=None, device=None):
    """BLEU-1~4（以及可選的 BERTScore）。hyps: list[str]；refs: list[list[str]]。"""
    out = compute_bleu(hyps, refs)
    if bertscore:
        out.update(compute_bertscore(hyps, refs, model_type=bertscore_model, device=device))
    return out


def latency_stats(seconds):
    """每張圖的回覆時間統計（毫秒）。"""
    ms = np.asarray(seconds, dtype=float) * 1000
    return {
        "Latency mean (ms)": float(ms.mean()),
        "Latency median (ms)": float(np.median(ms)),
        "Latency p95 (ms)": float(np.percentile(ms, 95)),
    }


# ----------------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------------
def get_gemini_client(api_key=None):
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        try:  # Colab 左側「密鑰」
            from google.colab import userdata

            key = userdata.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        from getpass import getpass

        key = getpass("請輸入 Gemini API key（輸入時不會顯示）：")
    try:
        from google import genai
    except ImportError as e:
        raise ImportError("使用 Gemini 需要 google-genai：pip install google-genai") from e
    return genai.Client(api_key=key)


def list_gemini_models(client):
    """列出可用來生成內容的模型名稱，模型名稱失效時用來查詢。"""
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if not actions or "generateContent" in actions:
            names.append(m.name.replace("models/", ""))
    return sorted(names)


def _hint(msg):
    """依錯誤訊息給一句白話的排查方向。"""
    m = str(msg).lower()
    if "404" in m or "not_found" in m or "not found" in m or "no longer available" in m:
        return "模型名稱不存在或已下架：改用 check_gemini 自動挑選，或用 list_gemini_models(client) 查目前可用的名稱。"
    if "api key" in m or "api_key" in m or "401" in m or "403" in m or "permission" in m or "unauthenticated" in m:
        return "API key 無效、沒有權限，或該專案沒有啟用 Gemini API。"
    if "429" in m or "resource_exhausted" in m or "quota" in m:
        return "超過額度或呼叫太頻繁：調大 gemini_sleep、減少 n_gemini，或稍後再試。"
    if "connect" in m or "timed out" in m or "timeout" in m or "proxy" in m or "ssl" in m:
        return "連不到 Gemini：檢查網路、防火牆或 proxy。"
    return ""


def _pick_flash(names):
    """從可用模型中挑一個穩定版的 flash 模型（排除 lite / preview / image / tts 等變體）。"""
    bad = ("lite", "image", "tts", "live", "audio", "embedding", "thinking", "exp", "preview",
           "robotics", "computer", "learnlm", "gemma", "vision", "native")
    cands = [n for n in names if "flash" in n and not any(b in n for b in bad)]
    if "gemini-flash-latest" in cands:
        return "gemini-flash-latest"
    return max(cands, default=None)


def check_gemini(client, model=DEFAULT_GEMINI_MODEL):
    """正式批次呼叫前，先確認 API key 與模型都能用，並回傳「實際可用」的模型名稱。

    原本指定的模型不在可用清單時（例如已下架），會自動改用一個可用的 flash 模型並印出提示。
    任何一步失敗都會直接丟出含真正錯誤訊息的例外，不會悄悄吞掉。
    """
    try:
        names = list_gemini_models(client)
    except Exception as e:
        raise RuntimeError(f"無法取得 Gemini 模型清單：{type(e).__name__}: {str(e)[:300]}\n提示：{_hint(e)}") from e
    model = model.replace("models/", "")
    chosen = model
    if names and model not in names:
        chosen = _pick_flash(names)
        if chosen is None:
            raise RuntimeError(f"找不到可用的 flash 模型。目前可用的模型：{names}")
        print(f"[提醒] 模型 {model!r} 不在可用清單中，自動改用 {chosen!r}")
    try:
        _, sec, _ = _generate(client, chosen, "Reply with the single word OK.", _config())
    except Exception as e:
        raise RuntimeError(f"呼叫 {chosen!r} 失敗：{type(e).__name__}: {str(e)[:300]}\n提示：{_hint(e)}") from e
    print(f"Gemini 連線正常：使用模型 {chosen}（測試呼叫 {sec:.2f}s）")
    return chosen


class _FailFast:
    """連續失敗太多次就停止整批，並把第一個錯誤完整顯示出來，避免花掉額度卻看不到原因。"""

    def __init__(self, limit):
        self.limit, self.consec, self.first = limit, 0, None

    def ok(self):
        self.consec = 0

    def fail(self, e):
        msg = f"{type(e).__name__}: {str(e)[:400]}"
        if self.first is None:
            self.first = msg
            print(f"\n[警告] 第一個錯誤：{msg}\n       {_hint(msg)}")
        self.consec += 1
        if self.limit and self.consec >= self.limit:
            raise RuntimeError(f"Gemini 連續失敗 {self.consec} 次，已停止以免浪費額度。\n"
                               f"第一個錯誤：{self.first}\n提示：{_hint(self.first)}") from e
        return msg[:200]


_RETRY_HINTS = ("429", "500", "502", "503", "504", "unavailable", "resource_exhausted", "deadline",
                "overloaded", "timed out", "timeout", "connection")


def _generate(client, model, contents, config, max_retries=5):
    """呼叫 Gemini；遇到限流 / 暫時性錯誤時指數退避重試。
    回傳 (response, 成功那一次呼叫的秒數, 重試次數)。計時不含等待重試的時間。"""
    retries = 0
    while True:
        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            return resp, time.perf_counter() - t0, retries
        except Exception as e:
            if retries >= max_retries or not any(h in str(e).lower() for h in _RETRY_HINTS):
                raise
            retries += 1
            time.sleep(min(60, 2**retries) + random.random())


def _image_part(path):
    from google.genai import types

    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def _config(json_mode=False, temperature=0.0, thinking_budget=None):
    from google.genai import types

    kw = dict(temperature=temperature)
    if json_mode:
        kw["response_mime_type"] = "application/json"
    if thinking_budget is not None:
        kw["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    return types.GenerateContentConfig(**kw)


def _parse_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


JUDGE_PROMPT = """You are evaluating an image captioning system.
Look at the image, then decide whether the caption below correctly describes it.

Caption: "{caption}"

Answer with JSON only, using exactly these keys:
{{"match": true or false, "score": integer 1-5, "reason": "one short sentence"}}

Rules:
- match is true only if the main subjects, their actions and the scene in the caption are consistent with the image, and the caption does not mention important things that are not in the image.
- score: 1 = completely wrong, 2 = mostly wrong, 3 = partly right, 4 = mostly right, 5 = accurate."""

CAPTION_PROMPT = ("Describe this image in one short sentence, in the style of a Flickr8k caption "
                  "(about 8 to 12 words, plain English, no quotes, no extra text).")


def gemini_judge(client, image_path, caption, model=DEFAULT_GEMINI_MODEL, thinking_budget=None):
    """請 Gemini 判斷 caption 是否符合影像內容。回傳 dict：match、score、reason、latency、retries。"""
    resp, sec, retries = _generate(
        client, model,
        [_image_part(image_path), JUDGE_PROMPT.format(caption=caption.replace('"', "'"))],
        _config(json_mode=True, thinking_budget=thinking_budget),
    )
    d = _parse_json(resp.text)
    score = int(round(float(d.get("score", 0))))
    return {"match": bool(d.get("match")), "score": min(5, max(1, score)),
            "reason": str(d.get("reason", "")).strip(), "latency": sec, "retries": retries}


def gemini_judge_batch(client, items, model=DEFAULT_GEMINI_MODEL, sleep=0.0, thinking_budget=None,
                       max_consecutive_errors=3):
    """items: [(image_path, caption), ...]。偶發的單張失敗不會中斷整批，錯誤記錄在 error 欄；
    但連續失敗 max_consecutive_errors 次（通常是 key / 模型 / 額度有問題）會立刻停止並顯示錯誤。"""
    rows, guard = [], _FailFast(max_consecutive_errors)
    for k, (path, cap) in enumerate(items, 1):
        row = {"image": Path(path).name, "caption": cap}
        try:
            row.update(gemini_judge(client, path, cap, model, thinking_budget))
            guard.ok()
        except Exception as e:
            row.update({"match": None, "score": None, "reason": "", "latency": None, "retries": None,
                        "error": guard.fail(e)})
        rows.append(row)
        print(f"\r[Gemini 判斷] {k}/{len(items)}", end="")
        if sleep:
            time.sleep(sleep)
    print()
    df = pd.DataFrame(rows)
    if "error" not in df:
        df["error"] = None
    return df


def summarize_judgements(df, label=""):
    ok = df[df["match"].notna()]
    return {
        "對象": label,
        "判斷張數": len(ok),
        "失敗張數": len(df) - len(ok),
        "符合比例": float(ok["match"].astype(bool).mean()) if len(ok) else float("nan"),
        "平均分數(1-5)": float(ok["score"].astype(float).mean()) if len(ok) else float("nan"),
        "平均耗時(s)": float(ok["latency"].astype(float).mean()) if len(ok) else float("nan"),
    }


def gemini_caption(client, image_path, model=DEFAULT_GEMINI_MODEL, thinking_budget=None):
    """讓 Gemini 直接替影像寫一句描述（zero-shot，作為對照方法）。回傳 (描述, 耗時秒數, 重試次數)。"""
    resp, sec, retries = _generate(
        client, model, [_image_part(image_path), CAPTION_PROMPT],
        _config(json_mode=False, temperature=0.0, thinking_budget=thinking_budget),
    )
    text = (resp.text or "").strip().strip('"').splitlines()[0] if (resp.text or "").strip() else ""
    return text, sec, retries


def gemini_caption_batch(client, paths, model=DEFAULT_GEMINI_MODEL, sleep=0.0, thinking_budget=None,
                         max_consecutive_errors=3):
    rows, guard = [], _FailFast(max_consecutive_errors)
    for k, path in enumerate(paths, 1):
        row = {"image": Path(path).name}
        try:
            cap, sec, retries = gemini_caption(client, path, model, thinking_budget)
            row.update({"caption": cap, "latency": sec, "retries": retries, "error": None})
            guard.ok()
        except Exception as e:
            row.update({"caption": "", "latency": None, "retries": None, "error": guard.fail(e)})
        rows.append(row)
        print(f"\r[Gemini 生成] {k}/{len(paths)}", end="")
        if sleep:
            time.sleep(sleep)
    print()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 繪圖
# ----------------------------------------------------------------------------
def plot_caption_history(histories, save_path=None):
    """histories: {名稱: train_caption 的 history}。左：loss；右：驗證集 perplexity。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        line, = axes[0].plot(ep, h["train_loss"], label=f"{name} train")
        axes[0].plot(ep, h["val_loss"], "--", color=line.get_color(), label=f"{name} val")
        axes[1].plot(ep, h["val_ppl"], marker="o", label=name)
    axes[0].set(title="Loss (per token)", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation perplexity (lower is better)", xlabel="Epoch", ylabel="Perplexity")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_caption_examples(images_dir, rows, ncols=3, title="", save_path=None, wrap=44):
    """rows: [{'image': 檔名, 'lines': [(標籤, 文字), ...]}]；每格顯示圖片與各方法的描述。"""
    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 5.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, r in zip(axes, rows):
        ax.imshow(np.asarray(Image.open(Path(images_dir) / r["image"]).convert("RGB")))
        text = "\n".join(textwrap.fill(f"{lab}: {txt}", wrap) for lab, txt in r["lines"])
        ax.set_title(text, fontsize=8, loc="left")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_method_comparison(df, quality="BLEU-4", latency="Latency mean (ms)", save_path=None):
    """左：各方法的品質分數；右：品質與回覆時間的取捨（時間用對數軸，Gemini 與本機模型差距很大）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    d = df.dropna(subset=[quality])
    colors = [plt.cm.tab10(i % 10) for i in range(len(d))]  # 兩張圖共用顏色，方法名稱放在圖例
    axes[0].barh(d.index, d[quality], color=colors)
    axes[0].set(title=quality, xlabel=quality)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)
    for i, v in enumerate(d[quality]):
        axes[0].text(v, i, f" {v:.3f}", va="center", fontsize=8)
    for (name, row), c in zip(d.iterrows(), colors):
        axes[1].scatter(row[latency], row[quality], s=70, color=c, label=name, edgecolor="black", linewidth=0.5)
    axes[1].set(xscale="log", xlabel=f"{latency} (log scale)", ylabel=quality, title="Quality vs response time")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()