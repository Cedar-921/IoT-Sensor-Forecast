"""src.models.baselines.ARIMAModel 的单元测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import ARIMAModel
from src.train import load_cleaned_data


@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    """复用真实清洗数据，整个模块共享。"""
    return load_cleaned_data()


def test_arima_fit_returns_self(cleaned_df):
    """fit() 应返回 self 以支持链式调用。"""
    model = ARIMAModel(order=(1, 0, 0))  # 用最简单的 order 加速测试
    y = cleaned_df["Appliances"].iloc[:200]
    result = model.fit(y)
    assert result is model
    assert model.model_ is not None


def test_arima_predict_shape_and_positive(cleaned_df):
    """predict() 返回正确形状的正值数组。"""
    model = ARIMAModel(order=(1, 0, 0))
    y = cleaned_df["Appliances"].iloc[:200]
    model.fit(y)
    preds = model.predict(n_steps=10)
    assert preds.shape == (10,)
    assert (preds > 0).all(), "ARIMA 预测必须为正（expm1）"


def test_arima_predict_before_fit_raises():
    """未 fit 就 predict 应抛 RuntimeError。"""
    model = ARIMAModel()
    with pytest.raises(RuntimeError) as excinfo:
        model.predict(n_steps=5)
    # 精确匹配错误信息（避免测试与生产代码语义耦合）
    assert excinfo.value.args[0] == "必须先调用 fit() 才能 predict"


def test_arima_save_load_roundtrip(tmp_path, cleaned_df):
    """save + load 后预测结果应一致。"""
    model = ARIMAModel(order=(1, 0, 0))
    y = cleaned_df["Appliances"].iloc[:200]
    model.fit(y)
    preds_before = model.predict(n_steps=10)

    p = tmp_path / "arima.pkl"
    model.save(p)
    assert p.exists()

    loaded = ARIMAModel.load(p)
    preds_after = loaded.predict(n_steps=10)
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)


def test_arima_insufficient_data_raises():
    """训练数据 < 50 应抛 ValueError。"""
    model = ARIMAModel()
    y = pd.Series([10.0] * 30)
    with pytest.raises(ValueError, match="训练数据太少"):
        model.fit(y)


def test_arima_predict_invalid_n_steps_raises(cleaned_df):
    """predict(n_steps<=0) 应抛 ValueError。"""
    model = ARIMAModel(order=(1, 0, 0))
    y = cleaned_df["Appliances"].iloc[:200]
    model.fit(y)
    with pytest.raises(ValueError, match="n_steps 必须 > 0"):
        model.predict(n_steps=0)
    with pytest.raises(ValueError, match="n_steps 必须 > 0"):
        model.predict(n_steps=-1)
