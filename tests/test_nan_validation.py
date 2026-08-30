"""所有模型 NaN/Inf 输入校验测试（C-5 修复）。

验证：
- X 含 NaN → ValueError
- y 含 NaN/Inf → ValueError（深度模型）
- y 含负值 → ValueError（XGBoost，log1p 域）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import XGBoostModel


def _make_features(n=100):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })


# ─────────────── XGBoost ───────────────

def test_xgboost_rejects_nan_in_X():
    """X 含 NaN 应抛 ValueError（C-5 修复）。"""
    rng = np.random.default_rng(0)
    X = _make_features(100)
    X.iloc[5, 1] = float("nan")
    y = pd.Series(np.abs(rng.normal(size=100) * 50 + 60))
    model = XGBoostModel()
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X, y)


def test_xgboost_rejects_negative_y():
    """y 含负值应抛 ValueError（M-2 修复）。"""
    X = _make_features(100)
    y = pd.Series([60.0] * 99 + [-1.0])  # 最后一个是负值
    model = XGBoostModel()
    with pytest.raises(ValueError, match="负值"):
        model.fit(X, y)


# ─────────────── LSTM / Transformer NaN 校验（依赖 torch）──────────────
# 先尝试 import torch；未装则 torch 相关测试 skip（XGBoost 测试不受影响）
try:
    import torch  # noqa: F401
    from src.models.lstm_model import LSTMModel
    from src.models.transformer_model import TransformerModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 注意：不用 pytestmark，让 XGBoost 测试始终跑，torch 相关测试在函数内 skip
requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="需要 torch（云端训练时跑）"
)


def _make_seq_features(n=120):
    """构造 LSTM/Transformer 用特征。"""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    }), pd.Series(np.abs(rng.normal(size=n) * 50 + 60))


@requires_torch
def test_lstm_rejects_nan_in_X():
    """LSTM X 含 NaN 应抛 ValueError（C-5）。"""
    X, y = _make_seq_features(120)
    X.iloc[10, 0] = float("nan")
    params = {
        "window_size": 4, "hidden_dim": 8, "num_layers": 1,
        "dropout": 0.0, "learning_rate": 1e-3, "batch_size": 16,
        "epochs": 1, "patience": 5, "grad_clip": 1.0, "random_state": 42,
    }
    model = LSTMModel(params=params, feature_names=["f1", "f2", "f3"])
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X, y)


@requires_torch
def test_lstm_rejects_nan_in_y():
    """LSTM y 含 NaN 应抛 ValueError（C-5）。"""
    X, y = _make_seq_features(120)
    y.iloc[5] = float("nan")
    params = {
        "window_size": 4, "hidden_dim": 8, "num_layers": 1,
        "dropout": 0.0, "learning_rate": 1e-3, "batch_size": 16,
        "epochs": 1, "patience": 5, "grad_clip": 1.0, "random_state": 42,
    }
    model = LSTMModel(params=params, feature_names=["f1", "f2", "f3"])
    with pytest.raises(ValueError, match="y 含 NaN"):
        model.fit(X, y)


@requires_torch
def test_transformer_rejects_inf_in_y():
    """Transformer y 含 Inf 应抛 ValueError（C-5）。"""
    X, y = _make_seq_features(120)
    y.iloc[3] = float("inf")
    params = {
        "window_size": 4, "hidden_dim": 8, "nhead": 2, "num_layers": 1,
        "dim_feedforward": 16, "dropout": 0.0, "learning_rate": 1e-3,
        "batch_size": 16, "epochs": 1, "patience": 5, "grad_clip": 1.0,
        "random_state": 42,
    }
    model = TransformerModel(params=params, feature_names=["f1", "f2", "f3"])
    with pytest.raises(ValueError, match="Inf"):
        model.fit(X, y)