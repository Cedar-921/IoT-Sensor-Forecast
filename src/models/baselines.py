"""传统时序基线模型：ARIMA。

内部使用 log1p 变换处理右偏目标，预测时用 expm1 反变换回原单位。
调用方无需关心 log 变换。
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA as _SMARIMA
from typing import Union


class ARIMAModel:
    """ARIMA 单变量时序预测（内部 log1p 变换）。"""

    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)):
        """初始化。

        参数
        ----------
        order : tuple[int, int, int]
            ARIMA 的 (p, d, q) 参数，默认 (2, 1, 2)。
        """
        self.order = order
        self.model_: _SMARIMA | None = None

    def fit(self, y_train: pd.Series) -> "ARIMAModel":
        """用训练数据拟合 ARIMA。

        参数
        ----------
        y_train : pd.Series
            单变量时间序列，**原始单位**（瓦）。

        返回
        -------
        ARIMAModel
            返回 self 以支持链式调用。
        """
        if len(y_train) < 50:
            raise ValueError(f"训练数据太少：{len(y_train)} < 50")
        y_log = np.log1p(y_train.astype(float))
        self.model_ = _SMARIMA(y_log, order=self.order).fit()
        return self

    def predict(self, n_steps: int) -> np.ndarray:
        """预测未来 n 步，自动反变换回原单位。

        参数
        ----------
        n_steps : int
            预测步数。

        返回
        -------
        np.ndarray
            形状 (n_steps,) 的预测值，原始单位（瓦）。
        """
        if self.model_ is None:
            raise RuntimeError("必须先调用 fit() 才能 predict")
        if n_steps <= 0:
            raise ValueError(f"n_steps 必须 > 0，实际 {n_steps}")
        forecast_log = self.model_.forecast(steps=n_steps)
        return np.expm1(forecast_log)

    def save(self, path: Union[str, Path]) -> None:
        """保存到磁盘，供 Flask API 加载。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model_, "order": self.order}, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ARIMAModel":
        """从磁盘恢复，返回完整实例。"""
        bundle = joblib.load(Path(path))
        obj = cls(order=bundle["order"])
        obj.model_ = bundle["model"]
        return obj
