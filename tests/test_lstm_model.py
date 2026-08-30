"""src.models.lstm_model.LSTMModel 的单元测试。

风格镜像 test_xgboost.py：7 个测试覆盖 fit/predict/save/load/异常路径。
单测用最小超参（hidden_dim=8, epochs=2, window=4）让 CPU 也能秒过。
部署到云端后跑，验证前请先 `pip install torch`。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lstm_model import LSTMModel
from src.train import load_cleaned_data


# ─────────────── Fixtures & Helpers ───────────────

@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    """复用真实清洗数据，整个模块共享（与 test_xgboost.py 一致）。"""
    return load_cleaned_data()


def _make_features(
    df: pd.DataFrame,
    n: int = 200,
    n_features: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """从 cleaned_df 切片，构造 LSTM 训练特征（用真实 Appliances 列）。"""
    sample = df.iloc[:n]
    rng = np.random.default_rng(seed=42)
    feat = pd.DataFrame({
        "Appliances_lag1": sample["Appliances"].shift(1).fillna(60.0),
        "lights": sample["lights"],
        "T1": sample["T1"],
        "T_out": sample["T_out"],
        "RH_out": sample["RH_out"],
    })
    feat.index = sample.index
    # 加几列噪声特征确保 n_features 可控
    for i in range(n_features - 5):
        feat[f"noise_{i}"] = rng.normal(size=len(sample))
    y = sample["Appliances"]  # 原始单位
    return feat, y


def _tiny_params() -> dict:
    """构造一个最小超参配置，让 CPU 也能秒过 fit。"""
    return {
        "window_size": 4,
        "hidden_dim": 8,
        "num_layers": 1,        # 单层（避免 LSTM dropout num_layers>1 限制）
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 2,
        "patience": 5,
        "grad_clip": 1.0,
        "random_state": 42,
    }


# ─────────────── 7 个测试 ───────────────

def test_lstm_fit_returns_self(cleaned_df):
    """fit() 应返回 self 以支持链式调用；model_ 应被填充。"""
    params = _tiny_params()
    model = LSTMModel(params=params, feature_names=[
        "Appliances_lag1", "lights", "T1", "T_out", "RH_out",
    ])
    X, y = _make_features(cleaned_df, n=200)
    result = model.fit(X, y)
    assert result is model
    assert model.model_ is not None
    assert model._scaler_mean is not None
    assert model._scaler_std is not None


def test_lstm_predict_shape_and_positive(cleaned_df):
    """predict() 返回 (n - window_size,) 形状的正值数组（expm1 保证）。"""
    params = _tiny_params()
    model = LSTMModel(params=params, feature_names=[
        "Appliances_lag1", "lights", "T1", "T_out", "RH_out",
    ])
    X, y = _make_features(cleaned_df, n=200)
    model.fit(X, y)

    # 取 X 的前 50 行做 predict
    X_pred = X.iloc[:50]
    preds = model.predict(X_pred)
    window = params["window_size"]
    assert preds.shape == (len(X_pred) - window,)
    assert (preds > 0).all(), "LSTM 预测必须为正（expm1 反变换）"


def test_lstm_predict_before_fit_raises():
    """未 fit 就 predict 应抛 RuntimeError（与 XGBoostModel 镜像）。"""
    model = LSTMModel()
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    with pytest.raises(RuntimeError) as excinfo:
        model.predict(X)
    # 精确匹配错误信息（避免测试与生产代码语义耦合）
    assert excinfo.value.args[0] == "必须先调用 fit() 才能 predict"


def test_lstm_save_load_roundtrip(tmp_path, cleaned_df):
    """save + load 后预测结果应一致（rtol=1e-5，FP32 精度上限）。"""
    params = _tiny_params()
    model = LSTMModel(params=params, feature_names=[
        "Appliances_lag1", "lights", "T1", "T_out", "RH_out",
    ])
    X, y = _make_features(cleaned_df, n=200)
    model.fit(X, y)

    # 推理前
    X_pred = X.iloc[:30]
    preds_before = model.predict(X_pred)

    # 保存
    p = tmp_path / "lstm.pt"
    model.save(p)
    assert p.exists()

    # 加载 + 推理
    loaded = LSTMModel.load(p)
    preds_after = loaded.predict(X_pred)
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)


def test_lstm_feature_names_aligned(cleaned_df):
    """用错列名/列序的 X 应抛 ValueError（防线上推理事故，镜像 XGBoost）。"""
    params = _tiny_params()
    model = LSTMModel(
        params=params,
        feature_names=["a", "b", "c", "d", "e"],
    )
    X = pd.DataFrame({
        "a": np.linspace(0, 1, 200),
        "b": np.linspace(0, 1, 200),
        "c": np.linspace(0, 1, 200),
        "d": np.linspace(0, 1, 200),
        "e": np.linspace(0, 1, 200),
    })
    y = pd.Series(np.abs(np.linspace(10, 50, 200)))
    model.fit(X, y)

    # 缺少训练时特征列 'a'
    bad = pd.DataFrame({
        "x": np.zeros(20), "y": np.zeros(20), "z": np.zeros(20),
    })
    with pytest.raises(ValueError, match="缺少"):
        model.predict(bad)

    # 含训练时未见特征列
    bad2 = pd.DataFrame({
        "a": np.zeros(20), "b": np.zeros(20), "c": np.zeros(20),
        "d": np.zeros(20), "e": np.zeros(20), "extra": np.zeros(20),
    })
    with pytest.raises(ValueError, match="未见"):
        model.predict(bad2)


def test_lstm_fit_empty_x_raises():
    """fit(X) 的 X 为空应抛 ValueError。"""
    model = LSTMModel()
    X_empty = pd.DataFrame(columns=["a", "b"])
    y_empty = pd.Series([], dtype=float)
    with pytest.raises(ValueError, match="训练数据为空"):
        model.fit(X_empty, y_empty)


def test_lstm_fit_length_mismatch_raises():
    """fit(X, y) 长度不一致应抛 ValueError。"""
    model = LSTMModel()
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([10.0, 20.0])  # 长度 2 vs 3
    with pytest.raises(ValueError, match="X/y 长度不一致"):
        model.fit(X, y)


# ─────────────── 额外验证：save 前未 fit 应报错 ───────────────

def test_lstm_save_before_fit_raises(tmp_path):
    """未 fit 就 save 应抛 RuntimeError（防线上事故）。"""
    model = LSTMModel()
    p = tmp_path / "lstm_unfit.pt"
    with pytest.raises(RuntimeError, match="fit"):
        model.save(p)


# ─────────────── 额外验证：早期 epoch 打印日志 ───────────────

def test_lstm_fit_uses_cuda_if_available(caplog, cleaned_df):
    """fit 应自动选 cuda/cpu（云端 GPU 训练前提）。"""
    import logging
    caplog.set_level(logging.INFO, logger="src.models.lstm_model")

    params = _tiny_params()
    model = LSTMModel(params=params, feature_names=[
        "Appliances_lag1", "lights", "T1", "T_out", "RH_out",
    ])
    X, y = _make_features(cleaned_df, n=200)
    model.fit(X, y)

    # 验证 device 字段类型正确（不验证具体值，因为环境相关）
    assert isinstance(model.device, torch.device)
    # 验证日志至少有一条 epoch 信息
    assert any("LSTM epoch" in rec.message for rec in caplog.records), (
        "应至少有一条 LSTM epoch 训练日志"
    )
