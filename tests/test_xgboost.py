"""src.models.xgboost_model.XGBoostModel 的单元测试。

风格镜像 test_arima.py：用真实 cleaned_df fixture 共享，5 个测试。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import XGBoostModel
from src.train import load_cleaned_data


@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    """复用真实清洗数据，整个模块共享。"""
    return load_cleaned_data()


def _make_features(df: pd.DataFrame, n: int = 300) -> tuple[pd.DataFrame, pd.Series]:
    """从 cleaned_df 切片，取最后 n 行做简单特征（直接用 df 列）。"""
    sample = df.iloc[:n]
    # 简单 5 维特征，足以让 XGBoost 训练并预测
    feat = pd.DataFrame({
        "Appliances_lag1": sample["Appliances"].shift(1).fillna(60),
        "lights": sample["lights"],
        "T1": sample["T1"],
        "T_out": sample["T_out"],
        "RH_out": sample["RH_out"],
    })
    feat.index = sample.index
    y = sample["Appliances"]  # 原始单位（XGBoostModel 内部 log1p）
    return feat, y


def test_xgboost_fit_returns_self(cleaned_df):
    """fit() 应返回 self 以支持链式调用。"""
    model = XGBoostModel(params={"n_estimators": 50, "max_depth": 3})
    X, y = _make_features(cleaned_df, n=200)
    result = model.fit(X, y)
    assert result is model
    assert model.model_ is not None


def test_xgboost_predict_shape_and_positive(cleaned_df):
    """predict() 返回正确形状的正值数组。"""
    model = XGBoostModel(params={"n_estimators": 50, "max_depth": 3})
    X, y = _make_features(cleaned_df, n=200)
    model.fit(X, y)
    preds = model.predict(X.iloc[:10])
    assert preds.shape == (10,)
    assert (preds > 0).all(), "XGBoost 预测必须为正（expm1）"


def test_xgboost_predict_before_fit_raises():
    """未 fit 就 predict 应抛 RuntimeError。"""
    model = XGBoostModel()
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    with pytest.raises(RuntimeError) as excinfo:
        model.predict(X)
    # 精确匹配错误信息（避免测试与生产代码语义耦合）
    assert excinfo.value.args[0] == "必须先调用 fit() 才能 predict"


def test_xgboost_save_load_roundtrip(tmp_path, cleaned_df):
    """save + load 后预测结果应一致。"""
    model = XGBoostModel(params={"n_estimators": 50, "max_depth": 3})
    X, y = _make_features(cleaned_df, n=200)
    model.fit(X, y)
    preds_before = model.predict(X.iloc[:10])

    p = tmp_path / "xgboost.pkl"
    model.save(p)
    assert p.exists()

    loaded = XGBoostModel.load(p)
    preds_after = loaded.predict(X.iloc[:10])
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)


def test_xgboost_feature_names_aligned(cleaned_df):
    """用错列名/列序的 X 应抛 ValueError（防线上推理事故）。"""
    model = XGBoostModel(
        params={"n_estimators": 50, "max_depth": 3},
        feature_names=["a", "b", "c"],
    )
    X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]})
    y = pd.Series([10.0])  # 原始单位
    model.fit(X, y)

    # 缺少训练时特征列 'a'
    bad = pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]})
    with pytest.raises(ValueError, match="缺少"):
        model.predict(bad)

    # 含训练时未见特征列
    bad2 = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0], "extra": [4.0]})
    with pytest.raises(ValueError, match="未见"):
        model.predict(bad2)


def test_xgboost_fit_empty_x_raises():
    """fit(X) 的 X 为空应抛 ValueError。"""
    model = XGBoostModel()
    X_empty = pd.DataFrame(columns=["a", "b"])
    y_empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="训练数据为空"):
        model.fit(X_empty, y_empty)


def test_xgboost_fit_length_mismatch_raises():
    """fit(X, y) 长度不一致应抛 ValueError。"""
    model = XGBoostModel()
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([10.0, 20.0])  # 长度 2 vs 3
    with pytest.raises(ValueError, match="X/y 长度不一致"):
        model.fit(X, y)