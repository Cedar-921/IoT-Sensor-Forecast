"""模型评估指标：MAE / RMSE / MAPE。

所有指标基于原始能耗单位（瓦），不接受 log 单位输入。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算 MAE / RMSE / MAPE 三个指标。

    参数
    ----------
    y_true : np.ndarray
        真实值，原始单位（瓦），shape (n,)
    y_pred : np.ndarray
        预测值，原始单位（瓦），shape (n,)

    返回
    -------
    dict[str, float]
        {"MAE": float, "RMSE": float, "MAPE": float}
        MAPE 单位为百分比（%），不是小数。

    异常
    ------
    ValueError
        当 y_true / y_pred 形状不一致或长度为 0 时。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致：y_true={y_true.shape}, y_pred={y_pred.shape}")
    if len(y_true) == 0:
        raise ValueError("输入数组为空")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE 防零保护（真实值接近 0 时）
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

    # sMAPE：对称 MAPE，输出范围 [0, 200]%，对零值友好
    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    # denominator 极小时（双方都接近 0）sMAPE → 0；直接除后按 mask 置 0，避免 warning
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    smape = float(np.mean(ratio) * 100)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}
