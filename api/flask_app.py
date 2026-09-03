"""Flask REST API 服务（**部署在云端**，供本地 Dashboard 调用）。

部署架构（方案 2：API 云 + Dashboard 本）：
    云端：本服务 + 训练好的模型 .pt / .pkl
    本地：dashboard/templates/index.html（静态 HTML），浏览器 fetch 调远程 API

云端启动：
    # 生产模式（推荐）
    gunicorn api.flask_app:app -b 0.0.0.0:5000 --workers 2

    # 开发模式（仅本地调试用）
    FLASK_DEBUG=1 python api/flask_app.py

本地 dashboard 启动（Git Bash）：
    python -m http.server 8080 --directory dashboard
    # 浏览器打开 http://localhost:8080

路由：
- /health 健康检查（k8s probe 用）
- /metrics 返回 reports/results.csv（所有模型的 MAE/RMSE/MAPE）
- /predict 接入模型推理（Day 16+，支持 arima/xgboost/lstm/transformer）

注意：
- 模型在第一次请求时懒加载（节省启动时间 + 显存）
- CORS 已配：允许 http://localhost:*（dashboard 本地浏览器 origin）
- 输入 schema 校验：api/schemas.py:PredictRequest（pydantic）
- 监听 0.0.0.0（云服务器必须，公网访问）
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# ─────────────── 路径（与 src/train.py 一致） ───────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.train import RESULTS_CSV  # noqa: E402

from src.feature_engineering import transform_features  # noqa: E402
from src.models.baselines import ARIMAModel  # noqa: E402
from src.models.xgboost_model import XGBoostModel  # noqa: E402

from api.schemas import PredictRequest, PredictResponse  # noqa: E402

# ─────────────── 日志 ───────────────
logger = logging.getLogger(__name__)

# ─────────────── Flask app + CORS ───────────────
app = Flask(__name__)

# 允许 dashboard 本地浏览器跨域调用（仅 8080 端口；演示场景专用）
# 生产环境若部署公网 HTTPS，可收紧 origins 为具体域名
# 注意：用具体 origin 而非通配符（http://localhost:*）防滥用
CORS(
    app,
    resources={r"/*": {"origins": [
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]}},
    supports_credentials=False,
)

# ─────────────── 模型懒加载（第一次请求时才加载） ───────────────
import threading  # noqa: E402

MODELS_DIR = _PROJECT_ROOT / "models"
CLEANED_CSV = _PROJECT_ROOT / "data" / "processed" / "cleaned.csv"

# 暖机数据行数 = max(lag_periods) = 288（用于 lag/rolling 计算时不 NaN）
WARMUP_ROWS = 288

_MODEL_LOADERS: dict[str, Any] = {
    "arima": ("arima_v1.pkl", "src.models.baselines", "ARIMAModel"),
    "xgboost": ("xgboost_v1.pkl", "src.models.xgboost_model", "XGBoostModel"),
    "lstm": ("lstm_v1.pt", "src.models.lstm_model", "LSTMModel"),
    "transformer": ("transformer_v1.pt", "src.models.transformer_model", "TransformerModel"),
}

# 模型缓存（避免每次请求都重新 load）
_model_cache: dict[str, Any] = {}
# 并发首次加载锁：gunicorn --threads N 时防止同一模型被重复加载
_model_lock = threading.Lock()

# cleaned.csv 末尾 WARMUP_ROWS 行的特征基线（懒加载）。
# 用于：当客户端只传 Appliances 单变量 history 时，
# 用最近的 cleaned.csv 行作为其他特征（T_out/RH/lights 等）的占位填值。
# 这是**演示场景简化**：生产环境应传完整特征矩阵。
_warmup_cache: pd.DataFrame | None = None
_warmup_lock = threading.Lock()


def _load_model(model_name: str) -> Any:
    """懒加载模型（线程安全，缓存到 _model_cache）。

    ARIMA / XGBoost: joblib.load
    LSTM / Transformer: torch.load（CPU 推理）

    注意：用 importlib 直接加载子模块，**不经过 src.models.__init__**，
    避免触发 torch 导入（本地无 torch 时也能加载 ARIMA/XGBoost）。
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    with _model_lock:
        # 双检（double-check）：加锁后再判一次，防 TOCTOU
        if model_name in _model_cache:
            return _model_cache[model_name]

        filename, module_name, class_name = _MODEL_LOADERS[model_name]
        model_path = MODELS_DIR / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_name} 模型文件不存在：{model_path}。"
                f"请先在云端跑：nohup python src/train.py --model {model_name}"
            )

        # importlib 标准 API（不覆盖 sys.modules，避免破坏已有引用）
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        model = cls.load(model_path)
        _model_cache[model_name] = model
        logger.info("已加载模型：%s (%s)", model_name, model_path)
        return model


def _load_warmup() -> pd.DataFrame:
    """加载 cleaned.csv 末尾 WARMUP_ROWS 行作为暖机数据（线程安全，缓存）。

    用于：当客户端只传 Appliances 单变量 history 时，
    用最近的 cleaned.csv 行的非目标特征（T_out/RH/lights 等）填占位。

    返回
    ------
    pd.DataFrame
        索引为 date（datetime），列含全部 27 个原始特征 + Appliances。
    """
    return _load_warmup_n_rows(WARMUP_ROWS)


def _load_warmup_n_rows(n_rows: int) -> pd.DataFrame:
    """加载 cleaned.csv 末尾 n_rows 行（线程安全，最大缓存）。

    比 WARMUP_ROWS 大的请求会重新读取（避免缓存浪费）。
    """
    global _warmup_cache
    if _warmup_cache is not None and len(_warmup_cache) >= n_rows:
        return _warmup_cache.tail(n_rows).copy()

    with _warmup_lock:
        if _warmup_cache is not None and len(_warmup_cache) >= n_rows:
            return _warmup_cache.tail(n_rows).copy()

        if not CLEANED_CSV.exists():
            raise FileNotFoundError(
                f"cleaned.csv 不存在：{CLEANED_CSV}。"
                f"请先在云端跑：python src/data_cleaning.py"
            )
        df = pd.read_csv(CLEANED_CSV, parse_dates=["date"]).set_index("date").sort_index()
        # 缓存取 max(已有, n_rows)
        target = max(n_rows, len(_warmup_cache) if _warmup_cache is not None else 0)
        _warmup_cache = df.iloc[-target:].copy() if target > 0 else df.iloc[0:0].copy()
        logger.info("已加载暖机数据：%d 行（%s → %s）",
                    len(_warmup_cache),
                    _warmup_cache.index[0], _warmup_cache.index[-1])
        return _warmup_cache.tail(n_rows).copy()


def _build_features_for_window(
    history: list[float],
    model_name: str,
    n_steps: int,
) -> pd.DataFrame:
    """用 history + cleaned.csv 暖机数据构造完整特征矩阵。

    策略：
    1. 暖机基线 = cleaned.csv 末尾 (WARMUP_ROWS + n_steps) 行
       — 保证 lag288 / rolling144 在每行都有真实历史可查
    2. 用 history 末尾 n_steps 个值替换暖机最后 n_steps 行的 Appliances
       — 客户端 history 是"过去 N 个 Appliances"，最后一个值是"现在"
    3. 调用 transform_features 构造 40+ 衍生特征
    4. 返回最后 n_steps 行（预测窗口）

    参数
    ----------
    history : list[float]
        客户端传入的 Appliances 历史序列（瓦）。至少 ≥ n_steps。
    model_name : str
        模型名（lstm / transformer 需要 window_size 信息，从模型对象取）。
    n_steps : int
        预测步数。

    返回
    ------
    pd.DataFrame
        含全部 40+ 衍生特征的 DataFrame，行数 = n_steps。
    """
    history_len = len(history)
    if history_len < n_steps:
        raise ValueError(
            f"history 长度不足：{history_len} < n_steps={n_steps}。"
            f"至少需要与预测步数等长的历史点"
        )

    # 1. 暖机基线：cleaned.csv 末尾 (WARMUP_ROWS + n_steps) 行
    full_warmup = _load_warmup_n_rows(WARMUP_ROWS + n_steps)
    warmup_modified = full_warmup.copy()

    # 2. 用 history 末尾 n_steps 个值替换暖机最后 n_steps 行的 Appliances
    #    先把整列转 float，避免 cleaned.csv 读出 int64 时赋值失败（LossySetitemError）
    warmup_modified["Appliances"] = warmup_modified["Appliances"].astype(float)
    warmup_modified["Appliances_log"] = warmup_modified["Appliances_log"].astype(float)
    target_history = np.asarray(history[-n_steps:], dtype=np.float64)
    warmup_modified.iloc[-n_steps:, warmup_modified.columns.get_loc("Appliances")] = target_history
    warmup_modified.iloc[-n_steps:, warmup_modified.columns.get_loc("Appliances_log")] = np.log1p(target_history)

    # 3. 构造特征
    feat = transform_features(warmup_modified, train_tail=warmup_modified.iloc[0:0])

    # 4. 取末尾 n_steps 行
    if len(feat) < n_steps:
        raise ValueError(
            f"暖机数据不足：构造后仅 {len(feat)} 行，少于 n_steps={n_steps}。"
            f"请检查 cleaned.csv"
        )
    feat = feat.iloc[-n_steps:].reset_index(drop=True)

    # 5. 校验 NaN（理论上不会，但防御性检查）
    if feat.isna().any().any():
        n_nan = int(feat.isna().sum().sum())
        cols = feat.columns[feat.isna().any()].tolist()
        raise ValueError(
            f"构造的特征含 {n_nan} 个 NaN，列={cols}（不应出现，请联系开发者）"
        )
    return feat


def _predict_with_history(
    model: Any,
    model_name: str,
    history: list[float],
    n_steps: int,
) -> list[float]:
    """统一处理 XGBoost / LSTM / Transformer 的 history → 推理。

    输入 history（list[float]）是 Appliances 历史值（瓦）。
    其他特征用 cleaned.csv 暖机数据填占位（演示简化）。

    返回 n_steps 个预测值（瓦）。

    内部用 _PREDICTORS 注册表按模型名分派（消除 if/elif 链）。
    """
    feat = _build_features_for_window(history, model_name, n_steps)

    # XGBoost / LSTM / Transformer 都需要严格对齐训练时的特征列
    train_feat_names = getattr(model, "feature_names", None)
    if train_feat_names is None:
        raise RuntimeError(f"{model_name} 模型缺少 feature_names，模型文件异常")

    missing = [c for c in train_feat_names if c not in feat.columns]
    if missing:
        raise ValueError(
            f"构造的特征缺训练时的列：{missing}。"
            f"通常是 history 长度不足（lag288 缺失）。请传 ≥ 290 个历史点。"
        )
    feat_aligned = feat[train_feat_names]

    predictor = _PREDICTORS.get(model_name)
    if predictor is None:
        raise NotImplementedError(f"未知模型：{model_name}")
    return predictor(model, feat_aligned, n_steps)


def _predict_xgboost(model: Any, feat_aligned: pd.DataFrame, n_steps: int) -> list[float]:
    """XGBoost：对每行做单点预测。"""
    preds = model.predict(feat_aligned)
    return [float(max(p, 0.0)) for p in preds[:n_steps]]


def _predict_seq_model(model: Any, feat_aligned: pd.DataFrame, n_steps: int) -> list[float]:
    """LSTM / Transformer：内部滑窗预测。"""
    window_size = model.params.get("window_size", 24)
    # predict 需要 len(X) >= window_size+1；不足时直接报错（客户端应传足够历史）
    if len(feat_aligned) < window_size + 1:
        raise ValueError(
            f"序列模型预测窗口不足：输入 {len(feat_aligned)} 行，"
            f"window_size={window_size}。请增加 history 长度"
        )
    preds_all = model.predict(feat_aligned)  # 长度 = n_input - window_size
    if len(preds_all) == 0:
        raise ValueError(
            f"预测结果为空：输入 {len(feat_aligned)} 行，window_size={window_size}"
        )
    return [float(max(p, 0.0)) for p in preds_all[-n_steps:]]


# 模型 → 预测策略注册表（消除 if/elif 链）
_PREDICTORS: dict[str, Any] = {
    "xgboost": _predict_xgboost,
    "lstm": _predict_seq_model,
    "transformer": _predict_seq_model,
}


# ─────────────── 路由 ───────────────

@app.route("/health", methods=["GET"])
def health():
    """健康检查端点（k8s probe / 监控用）。"""
    return jsonify({"status": "ok", "service": "iot-sensor-forecast"}), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """返回所有已训练模型的评估指标（来自 reports/results.csv，云端生成）。"""
    if not RESULTS_CSV.exists():
        return jsonify({"error": "results.csv 未生成，先在云端跑训练"}), 404
    df = pd.read_csv(RESULTS_CSV)
    return jsonify(df.to_dict(orient="records")), 200


@app.route("/history", methods=["GET"])
def history():
    """返回最近 N 个时间步的 Appliances 历史值（供 Dashboard 时序图用）。

    查询参数
    --------
    n : int, 默认 290
        返回的历史点数量（与训练窗口一致）

    返回
    ----
    JSON {"appliances": [float, ...], "n": int}
    """
    n = int(request.args.get("n", 290))
    if n <= 0 or n > 10000:
        return jsonify({"error": "n 必须在 1~10000 之间"}), 400
    try:
        df = _load_warmup_n_rows(n)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"appliances": df["Appliances"].tolist(), "n": len(df)}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """预测端点（Day 16+）。

    请求体（JSON）：
        {
            "model": "arima" | "xgboost" | "lstm" | "transformer",
            "n_steps": 1,                  # 预测步数（默认 1）
            "history": [60, 65, 70, ...]   # 历史 Appliances（XGB/LSTM/Transformer 必传）
        }

    返回：
        200 {"status":"ok", "model":"...", "predictions":[...], "metadata":{...}}
        400 {"status":"error", "error":"...", "detail":"..."}  # 入参不合法
        404 模型文件未找到
        500 内部错误
    """
    t_start = time.perf_counter()

    # 1. 解析 + 校验请求体
    payload = request.get_json(silent=True) or {}
    try:
        req = PredictRequest(**payload)
    except Exception as e:
        logger.warning("请求参数校验失败: %s", e)
        return jsonify({
            "status": "error",
            "error": "invalid_request",
            "detail": str(e),
        }), 400

    # 2. 加载模型（懒加载）
    try:
        model = _load_model(req.model)
    except FileNotFoundError as e:
        logger.warning("模型加载失败: %s", e)
        return jsonify({
            "status": "error",
            "error": "model_not_found",
            "detail": str(e),
        }), 404
    except ModuleNotFoundError as e:
        # 缺 torch 等依赖（本地 venv 未装 torch 时常见）
        logger.warning("模型依赖缺失: %s", e)
        return jsonify({
            "status": "error",
            "error": "missing_dependency",
            "detail": f"{e}；LSTM/Transformer 需要 torch，本地 venv 未安装。"
                      f"请在云端跑，或 pip install torch",
        }), 503
    except Exception as e:
        logger.exception("模型加载异常")
        return jsonify({
            "status": "error",
            "error": "model_load_failed",
            "detail": str(e),
        }), 500

    # 3. 调用模型推理
    try:
        if req.model == "arima":
            # ARIMA: 自带历史，只需 n_steps
            preds = model.predict(n_steps=req.n_steps)
        else:
            # XGBoost / LSTM / Transformer: 需要 history
            if req.history is None:
                raise ValueError(f"{req.model} 模型需要 history 字段")
            preds = _predict_with_history(
                model, req.model, req.history, req.n_steps,
            )
    except NotImplementedError as e:
        # 业务未实现（非内部错误），返回 501
        logger.info("predict not_implemented: model=%s, detail=%s", req.model, e)
        return jsonify({
            "status": "error",
            "error": "not_implemented",
            "detail": str(e),
        }), 501
    except Exception as e:
        logger.exception("推理失败")
        return jsonify({
            "status": "error",
            "error": "inference_failed",
            "detail": str(e),
        }), 500

    # 4. 构造响应
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    resp = PredictResponse(
        status="ok",
        model=req.model,
        predictions=[float(p) for p in preds],
        metadata={
            "n_steps": req.n_steps,
            "history_len": len(req.history) if req.history else 0,
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )
    logger.info(
        "predict OK: model=%s, n_steps=%d, elapsed=%.1fms",
        req.model, req.n_steps, elapsed_ms,
    )
    return jsonify(resp.model_dump()), 200


# ─────────────── 入口 ───────────────
if __name__ == "__main__":
    from src.logging_config import setup_logging
    setup_logging()
    # 仅本地调试用（生产请用 gunicorn，见模块 docstring）
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)