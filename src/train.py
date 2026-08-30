"""IoT-Sensor-Forecast · 统一训练入口（**云服务器专用**，本地不跑）。

训练一律跑在云服务器（Ubuntu + NVIDIA GPU），本地 Windows 只写代码 + 跑 pytest。

支持三种运行方式（仅云端）：

    # 方式 A（推荐）：作为模块运行
    python -m src.train --model arima

    # 方式 B（兼容）：直接运行脚本
    python src/train.py --model arima

    # 方式 C：显式指定项目根（非默认部署路径时）
    IOT_PROJECT_ROOT=/path/to/project python src/train.py --model arima

**必须用 `nohup ... &` 后台跑**，避免 SSH 断线杀掉训练进程：

    nohup python src/train.py --model lstm > logs/lstm_train.log 2>&1 &
    echo $! > logs/lstm_train.pid

详见 CLAUDE.md「训练环境（云端 GPU）」章节。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ─────────────── 直接运行兼容：把项目根加入 sys.path ───────────────
_PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FALLBACK))

# ─────────────── 项目根路径解析（云服务器可移植）───────────────
PROJECT_ROOT = Path(
    os.getenv("IOT_PROJECT_ROOT", _PROJECT_ROOT_FALLBACK)
)
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_CSV = REPORTS_DIR / "results.csv"

logger = logging.getLogger(__name__)


def time_split(
    df: pd.DataFrame, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """严格按时间切分（不随机）。"""
    n = len(df)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def load_cleaned_data() -> pd.DataFrame:
    """加载清洗后数据，时间排序。"""
    p = PROCESSED_DIR / "cleaned.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"cleaned.csv 不存在：{p}。先在云端跑：python src/data_cleaning.py"
        )
    return pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()


def append_results_csv(model_name: str, metrics: dict[str, float]) -> None:
    """追加一行到 results.csv（同名模型覆盖）。"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {"model": model_name, **metrics}
    if RESULTS_CSV.exists():
        existing = pd.read_csv(RESULTS_CSV)
        existing = existing[existing["model"] != model_name]
        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(RESULTS_CSV, index=False)
    logger.info("已写入 %s: %s", RESULTS_CSV.name, metrics)


def train_arima(n_train: int | None = None) -> dict[str, float]:
    """训练 ARIMA 并在验证集评估。

    参数
    ----------
    n_train : int | None
        限制训练样本数（用于快速 smoke test）。None = 全量数据。
        默认 None。

    返回
    -------
    dict[str, float]
        评估指标 {MAE, RMSE, MAPE}。
    """
    from src.models.baselines import ARIMAModel
    from src.evaluate import evaluate

    df = load_cleaned_data()
    train, val, test = time_split(df)

    if n_train is not None:
        train = train.iloc[:n_train]
        logger.info("ARIMA smoke test 模式：仅用 %d 条训练", n_train)

    y_train = train["Appliances"]
    y_val = val["Appliances"]

    logger.info("ARIMA: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(test))
    arima_order = (2, 1, 2)
    logger.info("ARIMA: order=%s, 预计拟合时间 1-5 分钟...", arima_order)

    model = ARIMAModel(order=arima_order)
    try:
        model.fit(y_train)
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.error("ARIMA 拟合失败: %s", e)
        # 返回 NaN 让上层不抛，但仍写 results.csv 留痕
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

    preds = model.predict(n_steps=len(y_val))
    metrics = evaluate(y_val.values, preds)

    # 保存模型（git 已排除 *.pkl）
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "arima_v1.pkl")
    logger.info("ARIMA 训练完成: %s", metrics)
    return metrics


def train_xgboost(n_train: int | None = None) -> dict[str, float]:
    """训练 XGBoost 并在验证集评估。

    参数
    ----------
    n_train : int | None
        限制训练样本数（用于快速 smoke test）。None = 全量数据。

    返回
    -------
    dict[str, float]
        评估指标 {MAE, RMSE, MAPE}。
    """
    from src.feature_engineering import (
        fit_transform_features,
        transform_features,
        drop_initial_nans,
    )
    from src.models.xgboost_model import XGBoostModel
    from src.evaluate import evaluate

    df = load_cleaned_data()
    train, val, test = time_split(df)

    if n_train is not None:
        train = train.iloc[:n_train]
        logger.info("XGBoost smoke test 模式：仅用 %d 条训练", n_train)

    # 特征工程（关键：train_tail 用于给 val/test 的 lag/rolling 暖机）
    train_feat, feature_cols, train_tail = fit_transform_features(train)
    val_feat = transform_features(val, train_tail)

    # 丢弃训练期首部 lag/rolling 引入的 NaN 行
    train_feat = drop_initial_nans(train_feat, feature_cols)

    y_train = train_feat["Appliances"]  # 原始单位（XGBoostModel 内部 log1p）
    X_train = train_feat[feature_cols]
    X_val = val_feat[feature_cols]

    logger.info(
        "XGBoost: train=%d (drop NaN 后), val=%d, features=%d",
        len(X_train), len(X_val), len(feature_cols),
    )

    model = XGBoostModel(feature_names=feature_cols)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    metrics = evaluate(val["Appliances"].values, preds)

    # 保存模型（git 已排除 *.pkl）
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "xgboost_v1.pkl")
    logger.info("XGBoost 训练完成: %s", metrics)
    return metrics


def train_lstm(n_train: int | None = None) -> dict[str, float]:
    """训练 LSTM 并在验证集评估。

    参数
    ----------
    n_train : int | None
        限制训练样本数（用于快速 smoke test）。

    返回
    -------
    dict[str, float]
        {MAE, RMSE, MAPE}。
    """
    from src.feature_engineering import (
        fit_transform_features,
        transform_features,
        drop_initial_nans,
    )
    from src.models.lstm_model import LSTMModel
    from src.evaluate import evaluate

    df = load_cleaned_data()
    train, val, test = time_split(df)

    if n_train is not None:
        train = train.iloc[:n_train]
        logger.info("LSTM smoke test 模式：仅用 %d 条训练", n_train)

    # 特征工程（与 train_xgboost 完全一致）
    train_feat, feature_cols, train_tail = fit_transform_features(train)
    val_feat = transform_features(val, train_tail)
    train_feat = drop_initial_nans(train_feat, feature_cols)

    y_train = train_feat["Appliances"]
    y_val_raw = val_feat["Appliances"]

    logger.info(
        "LSTM: train=%d, val=%d, features=%d",
        len(train_feat), len(val_feat), len(feature_cols),
    )

    # 训练（含早停 + checkpoint 落盘）
    model = LSTMModel(feature_names=feature_cols)
    model.fit(
        train_feat[feature_cols],
        y_train,
        X_val=val_feat[feature_cols],
        y_val=y_val_raw,
        checkpoint_dir=MODELS_DIR,  # 防 SSH 断线丢进度
    )

    # 推理 + 评估（注意：predict 返回 len - window_size 个值，对齐 y_val）
    preds = model.predict(val_feat[feature_cols])
    window = model.params["window_size"]
    y_val_aligned = y_val_raw.values[window:]
    if len(preds) == 0:
        logger.warning("LSTM 预测结果为空，跳过评估")
        metrics = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    else:
        metrics = evaluate(y_val_aligned, preds[: len(y_val_aligned)])

    # 保存模型（git 已排除 *.pt）
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "lstm_v1.pt")

    logger.info("LSTM 训练完成: %s", metrics)
    return metrics


def train_transformer(n_train: int | None = None) -> dict[str, float]:
    """训练 Transformer 并在验证集评估。

    参数
    ----------
    n_train : int | None
        限制训练样本数（用于快速 smoke test）。

    返回
    -------
    dict[str, float]
        {MAE, RMSE, MAPE}。
    """
    from src.feature_engineering import (
        fit_transform_features,
        transform_features,
        drop_initial_nans,
    )
    from src.models.transformer_model import TransformerModel
    from src.evaluate import evaluate

    df = load_cleaned_data()
    train, val, test = time_split(df)

    if n_train is not None:
        train = train.iloc[:n_train]
        logger.info("Transformer smoke test 模式：仅用 %d 条训练", n_train)

    # 特征工程（与 train_lstm 完全一致）
    train_feat, feature_cols, train_tail = fit_transform_features(train)
    val_feat = transform_features(val, train_tail)
    train_feat = drop_initial_nans(train_feat, feature_cols)

    y_train = train_feat["Appliances"]
    y_val_raw = val_feat["Appliances"]

    logger.info(
        "Transformer: train=%d, val=%d, features=%d",
        len(train_feat), len(val_feat), len(feature_cols),
    )

    # 训练（含早停 + checkpoint 落盘）
    model = TransformerModel(feature_names=feature_cols)
    model.fit(
        train_feat[feature_cols],
        y_train,
        X_val=val_feat[feature_cols],
        y_val=y_val_raw,
        checkpoint_dir=MODELS_DIR,  # 防 SSH 断线丢进度
    )

    # 推理 + 评估（注意：predict 返回 len - window_size 个值，对齐 y_val）
    preds = model.predict(val_feat[feature_cols])
    window = model.params["window_size"]
    y_val_aligned = y_val_raw.values[window:]
    if len(preds) == 0:
        logger.warning("Transformer 预测结果为空，跳过评估")
        metrics = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    else:
        metrics = evaluate(y_val_aligned, preds[: len(y_val_aligned)])

    # 保存模型（git 已排除 *.pt）
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "transformer_v1.pt")

    logger.info("Transformer 训练完成: %s", metrics)
    return metrics


# 模型注册表（Day 10-12 LSTM 已加入；Day 13-14 Transformer 已加入）
TRAINERS: dict[str, Callable] = {
    "arima": train_arima,
    "xgboost": train_xgboost,
    "lstm": train_lstm,
    "transformer": train_transformer,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="IoT-Sensor-Forecast 统一训练入口")
    parser.add_argument(
        "--model", required=True, choices=list(TRAINERS.keys()),
        help="模型名称：arima / xgboost / lstm / transformer",
    )
    parser.add_argument(
        "--n-train", type=int, default=None,
        help="训练样本数（默认全量，smoke test 可设 5000）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("项目根目录: %s", PROJECT_ROOT)

    trainer = TRAINERS[args.model]
    metrics = trainer(n_train=args.n_train) if "n_train" in trainer.__code__.co_varnames else trainer()
    append_results_csv(args.model, metrics)


if __name__ == "__main__":
    main()
