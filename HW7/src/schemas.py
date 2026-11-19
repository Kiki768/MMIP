"""作業一：結構化輸出 Spec（Pydantic v2）。

``RoutingDecision`` 是「分流節點」的唯一輸出契約，四種實驗（A~D）都必須產出
符合此 Spec 的 JSON，下游的對話管理器（作業三）只依賴此契約運作。
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import DEPARTMENT_IDS

DepartmentId = Literal[tuple(DEPARTMENT_IDS)]  # type: ignore[valid-type]


class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"
    angry = "angry"


class ImageType(str, Enum):
    none = "none"
    product_damage = "product_damage"
    receipt_invoice = "receipt_invoice"
    error_screenshot = "error_screenshot"
    package_label = "package_label"
    other = "other"


class SafetyFlag(str, Enum):
    safe = "safe"
    prompt_injection = "prompt_injection"
    jailbreak = "jailbreak"
    image_injection = "image_injection"
    pii_exposure = "pii_exposure"
    harmful = "harmful"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class ExtractedEntities(_Strict):
    order_id: Optional[str] = Field(None, description="訂單編號，例如 ORD-20261103-0042")
    product: Optional[str] = Field(None, description="商品名稱或型號")
    amount: Optional[float] = Field(None, description="金額（新台幣）")
    error_code: Optional[str] = Field(None, description="系統錯誤代碼，例如 E-1024")


class ImageAnalysis(_Strict):
    has_image: bool = Field(..., description="本輪是否附圖")
    image_type: ImageType = Field(..., description="圖片類型")
    description: str = Field(..., description="一句話描述圖片中與客服相關的內容")


class RoutingDecision(_Strict):
    """分流節點的結構化輸出。"""

    department: DepartmentId = Field(..., description="應轉接的部門 ID")  # type: ignore[valid-type]
    intent: str = Field(..., description="使用者意圖，snake_case，例如 refund_damaged_item")
    urgency: Urgency
    sentiment: Sentiment
    requires_human: bool = Field(..., description="是否需要真人介入")
    is_off_topic: bool = Field(..., description="本輪是否與客服業務無關")
    safety_flag: SafetyFlag
    confidence: float = Field(..., ge=0.0, le=1.0)
    entities: ExtractedEntities
    image_analysis: ImageAnalysis
    context_summary: str = Field(..., description="整合多輪脈絡後的一句話摘要")


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------
def routing_json_schema() -> dict:
    """回傳可直接送入 LLM 的 JSON Schema。"""
    return RoutingDecision.model_json_schema()


def validate_routing_json(text: str) -> tuple[Optional[RoutingDecision], Optional[str]]:
    """「嚴格」驗證：輸出必須整段就是合法 JSON 且符合 Spec，不做任何修補。

    回傳 (decision, error)；成功時 error 為 None。
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e.msg}"
    try:
        return RoutingDecision.model_validate(obj), None
    except ValidationError as e:
        first = e.errors()[0]
        return None, f"SchemaError: {'.'.join(map(str, first['loc']))} - {first['msg']}"


def lenient_parse(text: str) -> tuple[Optional[RoutingDecision], Optional[str]]:
    """「寬鬆」解析：嘗試去除 markdown code fence 與前後說明文字後再驗證（模擬常見的後處理修補）。"""
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            if t.lstrip().lower().startswith("json"):
                t = t.lstrip()[4:]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start : end + 1]
    return validate_routing_json(t)


def example_decision() -> RoutingDecision:
    """作業一 Spec 範例：使用者上傳破損商品照片並要求退貨。"""
    return RoutingDecision(
        department="returns_refunds",
        intent="return_damaged_item",
        urgency="high",
        sentiment="negative",
        requires_human=False,
        is_off_topic=False,
        safety_flag="safe",
        confidence=0.93,
        entities=ExtractedEntities(order_id="ORD-20261103-0042", product="SoundMax 藍牙耳機 X2", amount=2490.0),
        image_analysis=ImageAnalysis(
            has_image=True,
            image_type="product_damage",
            description="耳機充電盒外殼右側明顯裂痕，包裝盒角落凹陷",
        ),
        context_summary="使用者上一輪詢問到貨時間，本輪表示收到的耳機充電盒破裂，要求退貨並附上照片。",
    )
