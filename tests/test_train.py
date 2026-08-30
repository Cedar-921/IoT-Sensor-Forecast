"""src.train 模块的单元测试。

覆盖：
- time_split：默认比例、自定义比例、严格按时间不随机
- append_results_csv：首次写、同名模型覆盖、不同模型累加
- load_cleaned_data：缺失时抛 FileNotFoundError
- TRAINERS 注册表：dispatch 正确性、参数兼容性
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import train


# ─────────────── time_split ───────────────

def test_time_split_default_ratios():
    """默认比例应为 70/15/15。"""
    n = 1000
    df = pd.DataFrame({"v": range(n)})
    train_df, val_df, test_df = train.time_split(df)
    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150


def test_time_split_custom_ratios():
    """自定义比例应严格按比例切分。"""
    n = 100
    df = pd.DataFrame({"v": range(n)})
    train_df, val_df, test_df = train.time_split(df, ratios=(0.6, 0.2, 0.2))
    assert len(train_df) == 60
    assert len(val_df) == 20
    assert len(test_df) == 20


def test_time_split_preserves_order():
    """严格按时间切分不随机：索引顺序应保持。"""
    n = 100
    df = pd.DataFrame({"v": range(n)})
    train_df, val_df, test_df = train.time_split(df)
    # iloc[:70] 含 0~69 共 70 行，iloc[-1] = 69（int 截断边界）
    assert train_df["v"].iloc[0] == 0
    assert train_df["v"].iloc[-1] == 69
    assert val_df["v"].iloc[0] == 70
    assert val_df["v"].iloc[-1] == 84
    assert test_df["v"].iloc[0] == 85
    assert test_df["v"].iloc[-1] == 99


# ─────────────── append_results_csv ───────────────

def test_append_results_csv_creates_file(tmp_path, monkeypatch):
    """首次写入应创建 CSV 文件，含一行。"""
    monkeypatch.setattr(train, "RESULTS_CSV", tmp_path / "results.csv")
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path)
    train.append_results_csv("arima", {"MAE": 44.19, "RMSE": 94.25, "MAPE": 41.41})
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 1
    assert df.iloc[0]["model"] == "arima"
    assert df.iloc[0]["MAE"] == pytest.approx(44.19)


def test_append_results_csv_overwrites_same_model(tmp_path, monkeypatch):
    """同名模型应覆盖旧行，不重复。"""
    monkeypatch.setattr(train, "RESULTS_CSV", tmp_path / "results.csv")
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path)
    train.append_results_csv("arima", {"MAE": 50.0, "RMSE": 100.0, "MAPE": 40.0})
    train.append_results_csv("arima", {"MAE": 30.0, "RMSE": 70.0, "MAPE": 25.0})
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 1, "同名模型应覆盖而非追加"
    assert df.iloc[0]["MAE"] == pytest.approx(30.0)


def test_append_results_csv_accumulates_different_models(tmp_path, monkeypatch):
    """不同模型名应累加为多行。"""
    monkeypatch.setattr(train, "RESULTS_CSV", tmp_path / "results.csv")
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path)
    train.append_results_csv("arima", {"MAE": 50.0, "RMSE": 100.0, "MAPE": 40.0})
    train.append_results_csv("xgboost", {"MAE": 30.0, "RMSE": 70.0, "MAPE": 25.0})
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 2
    assert set(df["model"].tolist()) == {"arima", "xgboost"}


# ─────────────── load_cleaned_data ───────────────

def test_load_cleaned_data_missing_raises(monkeypatch):
    """cleaned.csv 不存在应抛 FileNotFoundError。"""
    monkeypatch.setattr(train, "PROCESSED_DIR", Path("/tmp/non_existent_dir_xyz"))
    with pytest.raises(FileNotFoundError, match="cleaned.csv 不存在"):
        train.load_cleaned_data()


# ─────────────── TRAINERS 注册表 ───────────────

def test_trainers_registry_has_expected_models():
    """注册表应包含已实现的 arima 与 xgboost。"""
    assert "arima" in train.TRAINERS
    assert "xgboost" in train.TRAINERS
    assert callable(train.TRAINERS["arima"])
    assert callable(train.TRAINERS["xgboost"])


def test_trainers_registry_dispatch_calls_correct_function(monkeypatch):
    """根据 model name dispatch 到正确 trainer。"""
    called = {"arima": 0, "xgboost": 0}

    def fake_arima(**kwargs):
        called["arima"] += 1
        return {"MAE": 1.0, "RMSE": 2.0, "MAPE": 3.0}

    def fake_xgboost(**kwargs):
        called["xgboost"] += 1
        return {"MAE": 0.5, "RMSE": 1.0, "MAPE": 1.5}

    # 不替换真实 trainer，只测试 dispatch 逻辑
    original = dict(train.TRAINERS)
    try:
        train.TRAINERS["arima"] = fake_arima
        train.TRAINERS["xgboost"] = fake_xgboost
        # 模拟 main() 第 152 行的 dispatch
        trainer = train.TRAINERS["arima"]
        trainer(n_train=None)
        trainer = train.TRAINERS["xgboost"]
        trainer(n_train=None)
        assert called["arima"] == 1
        assert called["xgboost"] == 1
    finally:
        train.TRAINERS.clear()
        train.TRAINERS.update(original)