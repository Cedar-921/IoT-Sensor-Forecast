"""XGBoost 多变量回归模型（带手工特征）。

内部使用 log1p 变换处理右偏目标，预测时用 expm1 反变换回原单位。
调用方无需关心 log 变换。API 与 src.models.baselines.ARIMAModel 对齐。
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union as _Union
from xgboost import XGBRegressor

DEFAULT_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


class XGBoostModel:
    """XGBoost 多变量时序预测（内部 log1p 变换）。"""

    def __init__(
        self,
        params: dict | None = None,
        feature_names: list[str] | None = None,
    ):
        """初始化。

        参数
        ----------
        params : dict | None
            覆盖默认超参。示例：{"n_estimators": 1000, "max_depth": 8}
        feature_names : list[str] | None
            训练用特征列名（按顺序）。predict 时若提供 X，将校验列名/顺序一致。
        """
        merged = {**DEFAULT_PARAMS, **(params or {})}
        self.params: dict = merged
        self.feature_names: list[str] | None = feature_names
        self.model_: XGBRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        """拟合 XGBoost 模型。

        参数
        ----------
        X : pd.DataFrame
            特征矩阵（shape (n_samples, n_features)）。
        y : pd.Series
            目标，**原始单位或 log 单位均可**（模型内部统一 log1p）。

        返回
        -------
        XGBoostModel
            返回 self 以支持链式调用。
        """
        if len(X) == 0:
            raise ValueError("训练数据为空")
        if len(X) != len(y):
            raise ValueError(f"X/y 长度不一致：X={len(X)}, y={len(y)}")

        # 输入无 NaN / Inf（XGBoost 内部静默 NaN）
        if X.isna().any().any():
            n = int(X.isna().sum().sum())
            cols = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"X 含 {n} 个 NaN，列={cols}；请检查上游清洗/特征工程"
            )

        # y 必须 >= 0（log1p 域），防止脏数据让 log1p=NaN
        y_arr = y.astype(float)
        if (y_arr < 0).any():
            n_neg = int((y_arr < 0).sum())
            raise ValueError(
                f"y 含 {n_neg} 个负值，log1p 域要求 y >= 0；请检查上游数据清洗"
            )

        # 记录特征名（如果之前没记录）
        if self.feature_names is None:
            self.feature_names = list(X.columns)

        # 强制列序对齐
        X_aligned = X[self.feature_names]

        y_log = np.log1p(y_arr)
        self.model_ = XGBRegressor(**self.params)
        self.model_.fit(X_aligned, y_log)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测，自动反变换回原单位。

        参数
        ----------
        X : pd.DataFrame
            特征矩阵，列名/列序应与训练时一致（否则抛 ValueError）。

        返回
        -------
        np.ndarray
            形状 (n_samples,)，原始单位（瓦）。
        """
        if self.model_ is None:
            raise RuntimeError("必须先调用 fit() 才能 predict")

        if self.feature_names is not None:
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(
                    f"X 缺少训练时的特征列：{missing}"
                )
            extra = [c for c in X.columns if c not in self.feature_names]
            if extra:
                raise ValueError(
                    f"X 含训练时未见的特征列：{extra}"
                )
            X_aligned = X[self.feature_names]
        else:
            X_aligned = X

        preds_log = self.model_.predict(X_aligned)
        return np.expm1(preds_log)

    def save(self, path: _Union[str, Path]) -> None:
        """保存到磁盘（含 booster / params / feature_names）。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model_,
                "params": self.params,
                "feature_names": self.feature_names,
            },
            p,
        )

    @classmethod
    def load(cls, path: _Union[str, Path]) -> "XGBoostModel":
        """从磁盘恢复，返回完整实例。"""
        bundle = joblib.load(Path(path))
        obj = cls(
            params=bundle.get("params"),
            feature_names=bundle.get("feature_names"),
        )
        obj.model_ = bundle["model"]
        return obj