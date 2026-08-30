"""src.evaluate 模块的单元测试。"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import evaluate


def test_evaluate_perfect_prediction():
    """完美预测：所有指标应为 0。"""
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    metrics = evaluate(y, y.copy())
    assert metrics["MAE"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["MAPE"] == pytest.approx(0.0, abs=1e-9)


def test_evaluate_constant_error():
    """常数误差：MAE == RMSE == 常数。"""
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 90.0, 110.0])
    metrics = evaluate(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(10.0, rel=1e-6)
    assert metrics["RMSE"] == pytest.approx(np.sqrt((100 + 100 + 100) / 3), rel=1e-6)
    assert metrics["MAPE"] == pytest.approx(10.0, rel=1e-6)  # |10/100|=10%


def test_evaluate_shape_mismatch_raises():
    """形状不一致应抛 ValueError。"""
    with pytest.raises(ValueError, match="形状不一致"):
        evaluate(np.array([1.0, 2.0]), np.array([1.0]))


def test_evaluate_empty_raises():
    """空数组应抛 ValueError。"""
    with pytest.raises(ValueError, match="输入数组为空"):
        evaluate(np.array([]), np.array([]))


def test_evaluate_zero_true_protection():
    """y_true 含 0 时 MAPE 不应为 inf（零保护）。"""
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 90.0, 210.0])
    metrics = evaluate(y_true, y_pred)
    # 第一项 |10-0|/max(|0|, 1e-9) ≈ 10/1e-9 会被截断为约 1e10 量级
    # 但 MAPE 仍应是有限 float，不应为 inf/nan
    assert np.isfinite(metrics["MAPE"]), f"MAPE 应有限，实际：{metrics['MAPE']}"
    # MAE / RMSE 不受零保护影响
    assert metrics["MAE"] == pytest.approx((10 + 10 + 10) / 3, rel=1e-6)
