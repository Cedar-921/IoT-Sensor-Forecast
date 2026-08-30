"""Flask API 的 Pydantic 输入/输出 Schema（Day 17）。

设计要点：
- 输入校验前置：避免把脏数据传到模型层（防御性编程）
- 支持 4 种模型的差异化请求体（ARIMA 单变量 vs XGB/LSTM/Transformer 滑窗）
- 输出统一封装 status / model / predictions / metadata
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ─────────────── 支持的模型清单 ───────────────
SUPPORTED_MODELS = ("arima", "xgboost", "lstm", "transformer")


# ─────────────── 基础请求 ───────────────

class PredictRequest(BaseModel):
    """通用预测请求基类。

    字段
    ----
    model : str
        模型名称，必须在 SUPPORTED_MODELS 中。
    n_steps : int
        预测步数（ARIMA 用），1~288（10min × 288 = 48h）。
    history : list[float] | None
        历史 Appliances 序列（瓦），XGBoost / LSTM / Transformer 用。
        ARIMA 不需要（自带历史）。
    n_features : int | None
        历史序列维度（多变量模型用），用于校验历史长度。
    """

    model: str = Field(..., description="模型名称")
    n_steps: int = Field(default=1, ge=1, le=288, description="预测步数")
    history: list[float] | None = Field(default=None, description="历史序列")
    n_features: int | None = Field(default=None, ge=1, description="特征维度")

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        v_lower = v.lower().strip()
        if v_lower not in SUPPORTED_MODELS:
            raise ValueError(
                f"不支持的 model={v!r}，"
                f"必须是 {SUPPORTED_MODELS} 之一"
            )
        return v_lower

    @field_validator("history")
    @classmethod
    def _validate_history(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("history 不能为空列表")
        if len(v) > 30_000:
            raise ValueError(
                f"history 太长（{len(v)} > 30000），请检查入参"
            )
        # 检查 NaN / Inf（pydantic 不自动校验浮点有效性）
        for i, x in enumerate(v):
            if not isinstance(x, (int, float)):
                raise ValueError(f"history[{i}]={x!r} 不是数值类型")
            if x != x:  # NaN
                raise ValueError(f"history[{i}] 是 NaN")
            if x in (float("inf"), float("-inf")):
                raise ValueError(f"history[{i}] 是 Inf")
        return [float(x) for x in v]


# ─────────────── 响应 ───────────────

class PredictResponse(BaseModel):
    """统一预测响应。

    字段
    ----
    status : str
        "ok" / "error"
    model : str
        实际推理的模型名。
    predictions : list[float]
        预测值（瓦），长度 = n_steps。
    metadata : dict
        附加元信息（训练集大小、特征数、推理耗时等）。
    """

    status: Literal["ok", "error"]
    model: str
    predictions: list[float]
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """错误响应（4xx / 5xx 时返回）。"""

    status: Literal["error"] = "error"
    error: str
    detail: str | None = None