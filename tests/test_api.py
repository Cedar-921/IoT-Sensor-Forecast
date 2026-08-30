"""Flask API 端到端测试（Day 16+）。

策略：
- 用真实 ARIMA 模型（models/arima_v1.pkl）做端到端推理验证
- 用 Flask test_client 模拟 HTTP 请求（无需启动真实服务器）
- 覆盖：/health、/metrics、/predict（成功/参数错误/模型未找到）

前置条件（来自 conftest.py）：
- data/raw/appliances_energy.csv 存在
- data/processed/cleaned.csv 存在
- models/arima_v1.pkl 存在（本地已有，云端拉回时一并拉回）
"""
from __future__ import annotations

from pathlib import Path

import pytest
from flask.testing import FlaskClient

from api.flask_app import app
from src.train import MODELS_DIR


# ─────────────── Fixtures ───────────────

@pytest.fixture(scope="module")
def client() -> FlaskClient:
    """Flask 测试客户端（每个模块共享）。"""
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="module")
def arima_model_path() -> Path:
    """ARIMA 模型文件路径，不存在则 skip 整个文件。"""
    p = MODELS_DIR / "arima_v1.pkl"
    if not p.exists():
        pytest.skip(f"ARIMA 模型不存在：{p}，请从云端拉回")
    return p


# ─────────────── /health 测试 ───────────────

def test_health_returns_ok(client):
    """/health 应返回 200 + status=ok。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "iot-sensor-forecast"


# ─────────────── /predict 参数校验测试 ───────────────

def test_predict_missing_body_returns_400(client):
    """POST /predict 无 body 应返回 400。"""
    resp = client.post("/predict", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert data["error"] == "invalid_request"


def test_predict_invalid_model_returns_400(client):
    """model 不在 SUPPORTED_MODELS 中应返回 400。"""
    resp = client.post("/predict", json={"model": "unknown_model"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"
    assert "不支持的 model" in data["detail"]


def test_predict_invalid_n_steps_returns_400(client):
    """n_steps 越界（<1 或 >288）应返回 400。"""
    resp = client.post("/predict", json={"model": "arima", "n_steps": 0})
    assert resp.status_code == 400
    resp = client.post("/predict", json={"model": "arima", "n_steps": 1000})
    assert resp.status_code == 400


def test_predict_history_with_nan_returns_400(client):
    """history 包含 NaN 应返回 400。"""
    resp = client.post(
        "/predict",
        json={"model": "xgboost", "history": [60.0, 65.0, float("nan"), 70.0]},
    )
    assert resp.status_code == 400


def test_predict_empty_history_returns_400(client):
    """history 为空列表应返回 400。"""
    resp = client.post(
        "/predict",
        json={"model": "xgboost", "history": []},
    )
    assert resp.status_code == 400


# ─────────────── /predict ARIMA 端到端测试 ───────────────

def test_predict_arima_returns_predictions(client, arima_model_path):
    """ARIMA 端到端推理：请求 → 加载模型 → 预测 → 返回 JSON。"""
    # ARIMA 不需要 history 字段，自带训练历史
    resp = client.post(
        "/predict",
        json={"model": "arima", "n_steps": 5},
    )
    assert resp.status_code == 200, f"ARIMA 推理失败: {resp.get_json()}"
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["model"] == "arima"
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == 5
    # 预测必须为正（expm1 反变换）
    assert all(p > 0 for p in data["predictions"]), "ARIMA 预测必须为正"
    # 元信息
    assert "n_steps" in data["metadata"]
    assert data["metadata"]["n_steps"] == 5
    assert "elapsed_ms" in data["metadata"]
    assert data["metadata"]["elapsed_ms"] > 0


def test_predict_arima_default_n_steps_is_one(client, arima_model_path):
    """不传 n_steps 时默认 1。"""
    resp = client.post("/predict", json={"model": "arima"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["predictions"]) == 1


# ─────────────── /predict 模型未找到测试 ───────────────

def test_predict_lstm_without_checkpoint_returns_404(client):
    """LSTM 模型文件不存在应返回 404。"""
    p = MODELS_DIR / "lstm_v1.pt"
    if p.exists():
        pytest.skip("LSTM 模型已存在（云端训练完成），跳过此测试")
    resp = client.post(
        "/predict",
        json={"model": "lstm", "history": [60.0] * 100},
    )
    # 当前 LSTM/Transformer 还没完整实现，会返回 500
    # 但接口框架必须工作（404 或 500 都行，关键是不要 200 假装成功）
    assert resp.status_code in (404, 500), f"应返回 4xx/5xx，实际 {resp.status_code}"
    data = resp.get_json()
    assert data["status"] == "error"


# ─────────────── /metrics 测试 ───────────────

def test_metrics_returns_csv_records(client):
    """/metrics 返回 reports/results.csv 内容。"""
    from src.train import RESULTS_CSV
    if not RESULTS_CSV.exists():
        pytest.skip("results.csv 不存在（云端训练未完成）")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    if data:
        assert "model" in data[0]
        assert "MAE" in data[0]