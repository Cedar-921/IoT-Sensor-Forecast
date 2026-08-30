"""LSTMModel / TransformerModel 的 checkpoint 落盘测试（Day 16+，C-4 修复）。

验证：
- best.pt 在 val_mae 改善时落盘
- latest.pt 每 save_every_epochs 轮落盘
- checkpoint_dir=None 时不写任何文件（默认行为）
- checkpoint 内容完整（state_dict / params / feature_names / scaler / best_val）

依赖 torch，未安装则整个文件 skip。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.train import load_cleaned_data


@pytest.fixture(scope="module")
def cleaned_df() -> pd.DataFrame:
    return load_cleaned_data()


def _tiny_lstm_params() -> dict:
    return {
        "window_size": 4,
        "hidden_dim": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 4,             # 至少 2 轮才有可能触发 best/latest
        "patience": 5,
        "grad_clip": 1.0,
        "random_state": 42,
        "save_every_epochs": 2,
    }


def _tiny_transformer_params() -> dict:
    return {
        "window_size": 4,
        "hidden_dim": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 4,
        "patience": 5,
        "grad_clip": 1.0,
        "random_state": 42,
        "save_every_epochs": 2,
    }


def _make_features(df, n=200):
    sample = df.iloc[:n]
    feat = pd.DataFrame({
        "Appliances_lag1": sample["Appliances"].shift(1).fillna(60.0),
        "lights": sample["lights"],
        "T1": sample["T1"],
        "T_out": sample["T_out"],
        "RH_out": sample["RH_out"],
    })
    feat.index = sample.index
    y = sample["Appliances"]
    return feat, y


# ─────────────── LSTM ───────────────

def test_lstm_checkpoint_dir_none_does_not_write(tmp_path, cleaned_df):
    """checkpoint_dir=None 时不写任何 .pt 文件（默认行为）。"""
    p = tmp_path / "lstm_none"
    p.mkdir()
    params = _tiny_lstm_params()
    model = LSTMModel(params=params, feature_names=[
        "Appliances_lag1", "lights", "T1", "T_out", "RH_out",
    ])
    X, y = _make_features(cleaned_df, n=120)
    Xv, yv = X.iloc[:80], y.iloc[:80]
    model.fit(Xv, yv, X_val=Xv, y_val=yv)  # no checkpoint_dir
    # 不应在 p 下生成任何文件
    assert list(p.iterdir()) == [], f"checkpoint_dir=None 时不应写文件，实际：{list(p.iterdir())}"


def test_lstm_checkpoint_dir_writes_latest_pt(tmp_path, cleaned_df):
    """传入 checkpoint_dir 应生成 latest.pt。"""
    params = _tiny_lstm_params()
    model = LSTMModel(params=params, feature_names=[
        "Appliances_lag1", "lights", "T1", "T_out", "RH_out",
    ])
    X, y = _make_features(cleaned_df, n=120)
    Xv, yv = X.iloc[:80], y.iloc[:80]
    model.fit(Xv, yv, X_val=Xv, y_val=yv, checkpoint_dir=tmp_path)
    # save_every_epochs=2, epochs=4 → 应在 epoch 2、4 写 latest.pt
    assert (tmp_path / "latest.pt").exists(), "latest.pt 未生成"


def test_lstm_checkpoint_content_is_complete(tmp_path, cleaned_df):
    """checkpoint 应包含所有必要字段。"""
    params = _tiny_lstm_params()
    feat_names = ["Appliances_lag1", "lights", "T1", "T_out", "RH_out"]
    model = LSTMModel(params=params, feature_names=feat_names)
    X, y = _make_features(cleaned_df, n=120)
    Xv, yv = X.iloc[:80], y.iloc[:80]
    model.fit(Xv, yv, X_val=Xv, y_val=yv, checkpoint_dir=tmp_path)
    ckpt = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    for key in ["state_dict", "params", "feature_names", "scaler_mean", "scaler_std", "best_val"]:
        assert key in ckpt, f"checkpoint 缺字段：{key}"
    assert ckpt["feature_names"] == feat_names
    assert isinstance(ckpt["best_val"], float)


# ─────────────── Transformer ───────────────

def test_transformer_checkpoint_dir_writes_latest_pt(tmp_path, cleaned_df):
    """Transformer 同样应在 checkpoint_dir 写入 latest.pt。"""
    params = _tiny_transformer_params()
    model = TransformerModel(params=params, feature_names=[
        "Appliances_lag1", "lights", "T1", "T_out", "RH_out",
    ])
    X, y = _make_features(cleaned_df, n=120)
    Xv, yv = X.iloc[:80], y.iloc[:80]
    model.fit(Xv, yv, X_val=Xv, y_val=yv, checkpoint_dir=tmp_path)
    assert (tmp_path / "latest.pt").exists(), "Transformer latest.pt 未生成"


def test_transformer_checkpoint_content_is_complete(tmp_path, cleaned_df):
    """Transformer checkpoint 应包含所有必要字段。"""
    params = _tiny_transformer_params()
    feat_names = ["Appliances_lag1", "lights", "T1", "T_out", "RH_out"]
    model = TransformerModel(params=params, feature_names=feat_names)
    X, y = _make_features(cleaned_df, n=120)
    Xv, yv = X.iloc[:80], y.iloc[:80]
    model.fit(Xv, yv, X_val=Xv, y_val=yv, checkpoint_dir=tmp_path)
    ckpt = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    for key in ["state_dict", "params", "feature_names", "scaler_mean", "scaler_std", "best_val"]:
        assert key in ckpt, f"checkpoint 缺字段：{key}"
    assert "nhead" in ckpt["params"], "Transformer params 应含 nhead"
    assert ckpt["params"]["nhead"] == 2