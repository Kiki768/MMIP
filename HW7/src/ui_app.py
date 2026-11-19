"""作業三：Gradio 多模態客服聊天室。

啟動方式（於 HW7/ 目錄）::

    python -m src.ui_app            # 預設 http://127.0.0.1:7860
    CHATBOT_LLM_BACKEND=claude python -m src.ui_app   # 強制使用 Claude（需 ANTHROPIC_API_KEY）

介面功能
* 文字 + 圖片（拖曳/貼上/上傳）同一輸入框送出
* 快速回覆按鈕（依對話狀態動態變化，降低打字成本）
* 側欄即時顯示：結構化分流 JSON、對話狀態（話題堆疊/已知槽位）、Guardrail 旗標與事件追蹤
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

from .chatbot_core import ChatbotCore, build_default_router
from .routing_engines import ImagePerception, resolve_backend

N_QUICK = 3
_SHARED: dict[str, Any] = {}


def _shared_components():
    """路由器（含已訓練的 System One）在所有使用者 session 之間共用，只訓練一次。"""
    if "router" not in _SHARED:
        perception = ImagePerception()
        _SHARED["perception"] = perception
        _SHARED["router"] = build_default_router(perception)
    return _SHARED["router"], _SHARED["perception"]


def _new_bot() -> ChatbotCore:
    router, perception = _shared_components()
    return ChatbotCore(router=router, perception=perception)


def _file_path(f: Any) -> Optional[str]:
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get("path") or f.get("name")
    return getattr(f, "path", None) or getattr(f, "name", None)


def _image_to_b64(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return base64.b64encode(Path(path).read_bytes()).decode()


def build_app():
    import gradio as gr

    major = int(gr.__version__.split(".")[0])
    chatbot_kwargs = dict(label="客服對話", height=540)
    if major < 6:
        chatbot_kwargs["type"] = "messages"

    with gr.Blocks(title="多模態客服機器人") as demo:
        gr.Markdown(f"## 🛍️ 多模態多輪客服機器人\n後端：`{resolve_backend()}`　｜　路由：System One → System Two 串接")
        bot_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(**chatbot_kwargs)
                box = gr.MultimodalTextbox(placeholder="輸入訊息，或拖曳/貼上商品照片、錯誤截圖、帳單…",
                                           file_types=["image"], file_count="single", show_label=False)
                with gr.Row():
                    quick = [gr.Button(visible=False, size="sm") for _ in range(N_QUICK)]
                reset = gr.Button("🔄 開新對話", variant="secondary", size="sm")
            with gr.Column(scale=2):
                routing_json = gr.JSON(label="結構化分流結果（RoutingDecision）")
                state_json = gr.JSON(label="對話狀態（話題堆疊 / 已知槽位）")
                trace_md = gr.Markdown("**Guardrail / 事件追蹤**：尚無")

        def respond(message: dict, history: list, bot: Optional[ChatbotCore]):
            bot = bot or _new_bot()
            text = (message or {}).get("text", "") or ""
            files = (message or {}).get("files", []) or []
            path = _file_path(files[0]) if files else None
            reply = bot.handle(text, _image_to_b64(path))
            history = list(history or [])
            if path:
                history.append({"role": "user", "content": {"path": path}})
            if text:
                history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": reply.text})
            flags = "、".join(reply.guard_flags) or "無"
            events = "\n".join(f"- `{e[:160]}`" for e in reply.events) or "- （無）"
            trace = f"**Guardrail 旗標**：{flags}\n\n**事件**（{reply.latency_s:.3f}s）：\n{events}"
            btn_updates = [gr.update(value=q, visible=True) for q in reply.quick_replies[:N_QUICK]]
            btn_updates += [gr.update(visible=False)] * (N_QUICK - len(btn_updates))
            return (history, gr.update(value=None), bot, reply.routing, reply.state, trace, *btn_updates)

        outputs = [chat, box, bot_state, routing_json, state_json, trace_md, *quick]
        box.submit(respond, [box, chat, bot_state], outputs)
        for b in quick:
            b.click(lambda label, h, s: respond({"text": label, "files": []}, h, s), [b, chat, bot_state], outputs)

        def do_reset():
            return ([], None, None, None, "**Guardrail / 事件追蹤**：尚無",
                    *[gr.update(visible=False)] * N_QUICK)

        reset.click(do_reset, None, [chat, bot_state, routing_json, state_json, trace_md, *quick])
    return demo


def main() -> None:
    _shared_components()
    build_app().launch()


if __name__ == "__main__":
    main()
