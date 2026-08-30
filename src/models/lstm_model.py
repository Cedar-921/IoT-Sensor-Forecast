"""IoT-Sensor-Forecast · LSTM 时序预测模型。

API 镜像 src.models.transformer_model.TransformerModel：
- fit(X, y) / fit(X, y, X_val, y_val)
- predict(X) -> np.ndarray（原始单位）
- save(path) / load(path) classmethod

内部细节（调用方不感知）：
- log1p(y) / expm1(pred) 与 XGBoost 对齐
- 标准化器（按列 mean/std）持久化在 checkpoint
- device 自动选 cuda/cpu
- 双层 LSTM + dropout + 早停

云端训练约定（详见 CLAUDE.md「训练环境（云端 GPU）」）：
- 云服务器 pip install torch --index-url .../whl/cu118（按 CUDA 版本）
- nohup python src/train.py --model lstm > logs/lstm.log 2>&1 &
- checkpoint 保存到 models/lstm_v1.pt（git 已忽略）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.sequence_dataset import SequenceDataset

logger = logging.getLogger(__name__)


# ─────────────── PyTorch 模型定义（私有） ───────────────

class _LSTMNet(nn.Module):
    """双层 LSTM + 线性头。

    结构：Input(batch, window_size, n_features)
        → LSTM(num_layers=2, hidden_dim, dropout)
        → 取最后时间步 → Linear(hidden → 1) → squeeze
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # 单层时不传 dropout（PyTorch LSTM 要求 num_layers>1 才能设 dropout）
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]               # (batch, hidden_dim)
        return self.head(last).squeeze(-1)  # (batch,)


# ─────────────── 公开 LSTMModel ───────────────

class LSTMModel:
    """LSTM 时序预测模型（包装类，调用方不接触 torch 细节）。"""

    DEFAULT_PARAMS: dict = {
        "window_size": 24,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 30,
        "patience": 5,
        "grad_clip": 1.0,
        "random_state": 42,
        "save_every_epochs": 5,   # checkpoint 落盘频率（仅在 checkpoint_dir 不为 None 时生效）
    }

    def __init__(
        self,
        params: dict | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """构造 LSTMModel。

        参数
        ----------
        params : dict | None
            超参覆盖；不传则用 DEFAULT_PARAMS。
        feature_names : list[str] | None
            训练时特征列名（用于 save/load 校验）。
        """
        self.params: dict = {**self.DEFAULT_PARAMS, **(params or {})}
        self.feature_names = feature_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_: _LSTMNet | None = None
        # 标准化器在 fit 时填，save/load 时持久化
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None

    # ──────────── 工具方法 ────────────

    def _validate_feature_names(self, X: pd.DataFrame) -> pd.DataFrame:
        """校验 X 列名与训练时一致，返回按训练列序重排的 X。

        不一致时抛 ValueError（防线上推理列错位事故）。
        """
        if self.feature_names is None:
            raise RuntimeError("必须先调用 fit() 才能 predict")
        missing = [c for c in self.feature_names if c not in X.columns]
        extra = [c for c in X.columns if c not in self.feature_names]
        if missing:
            raise ValueError(f"X 缺少训练时的特征列：{missing}")
        if extra:
            raise ValueError(f"X 含训练时未见的特征列：{extra}")
        return X[self.feature_names]

    @staticmethod
    def _validate_xy(X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """校验 fit/predict 入参的形状和长度。"""
        if X is None or len(X) == 0:
            raise ValueError("训练数据为空，请检查入参")
        if y is not None and len(X) != len(y):
            raise ValueError(
                f"X/y 长度不一致：{len(X)} vs {len(y)}"
            )

    @staticmethod
    def _validate_no_nan(X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """校验输入无 NaN / Inf，否则训练会静默输出 NaN（PyTorch）。"""
        if X.isna().any().any():
            n = int(X.isna().sum().sum())
            cols = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"X 含 {n} 个 NaN，列={cols}；请检查上游清洗/特征工程"
            )
        if y is not None:
            y_arr = y.astype(float).values
            if np.isnan(y_arr).any():
                raise ValueError("y 含 NaN；请检查上游清洗/特征工程")
            if np.isinf(y_arr).any():
                raise ValueError("y 含 Inf；请检查上游清洗/特征工程")

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        """用训练时 mean/std 标准化（按列）。"""
        if self._scaler_mean is None or self._scaler_std is None:
            raise RuntimeError("必须先调用 fit() 才能 predict")
        return (X.values - self._scaler_mean) / self._scaler_std

    def _build_loader(
        self,
        X_scaled: pd.DataFrame,
        y_log: np.ndarray,
        window: int,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """构造 DataLoader（统一 train / val / predict 三处入口）。"""
        ds = SequenceDataset(
            X_scaled,
            pd.Series(y_log),
            window_size=window,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,           # Windows 安全；云端 Linux 可手动改 2-4
            pin_memory=(self.device.type == "cuda"),
        )

    def _compute_val_mae(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        window: int,
    ) -> float:
        """在验证集上评估 log1p 空间 MAE（与 train loss 同空间）。"""
        X_val_aligned = self._validate_feature_names(X_val)
        y_val_log = np.log1p(y_val.astype(np.float32).values)
        X_val_scaled = pd.DataFrame(
            self._scale(X_val_aligned),
            columns=X_val_aligned.columns,
            index=X_val_aligned.index,
        )
        val_loader = self._build_loader(
            X_val_scaled, y_val_log, window,
            batch_size=self.params["batch_size"] * 2,
            shuffle=False,
        )
        self.model_.eval()
        preds_log: list[np.ndarray] = []
        with torch.no_grad():
            for Xb, _ in val_loader:
                preds_log.append(self.model_(Xb.to(self.device)).cpu().numpy())
        if not preds_log:
            return float("nan")
        val_pred_log = np.concatenate(preds_log)
        # 对齐：valid 区间从 window 开始
        val_y_aligned = y_val_log[window:]
        return float(np.mean(np.abs(val_pred_log[: len(val_y_aligned)] - val_y_aligned)))

    # ──────────── 训练 ────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> "LSTMModel":
        """训练 LSTM。

        参数
        ----------
        X : DataFrame
            训练特征矩阵（n, n_features），来自 fit_transform_features + dropna。
        y : Series
            训练目标（n,），原始单位 Appliances。
        X_val, y_val : 可选
            验证集；若提供则用 val_mae 做 early stopping。
        checkpoint_dir : str | Path | None
            检查点保存目录（云端防 SSH 断线丢进度）。
            - best.pt：val_mae 改善时覆盖
            - latest.pt：每 save_every_epochs 轮覆盖
            None = 不保存（默认；显式传路径才启用，避免本地 CPU 调试写一堆文件）。

        返回
        -------
        LSTMModel
            self（链式调用）。
        """
        self._validate_xy(X, y)
        self._validate_no_nan(X, y)

        # 1. 标准化（fit on X）
        self._scaler_mean = X.mean(axis=0).values.astype(np.float32)
        std = X.std(axis=0).values.astype(np.float32)
        # 防零方差（极端情况：某列全相等）
        std = np.where(std < 1e-9, 1.0, std)
        self._scaler_std = std

        # 2. log1p(y)
        y_log = np.log1p(y.astype(np.float32).values)

        # 3. 标准化 X（避免修改入参）
        X_scaled = pd.DataFrame(
            self._scale(X),
            columns=X.columns,
            index=X.index,
        )

        # 4. SequenceDataset + DataLoader
        window = self.params["window_size"]
        train_loader = self._build_loader(
            X_scaled, y_log, window,
            batch_size=self.params["batch_size"],
            shuffle=True,
        )

        # 5. 构造 PyTorch 模型
        torch.manual_seed(self.params["random_state"])
        # cuDNN 非确定性可能导致同样 seed 不同结果；显式启用确定性算法（M-4 修复）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model_ = _LSTMNet(
            n_features=X_scaled.shape[1],
            hidden_dim=self.params["hidden_dim"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.params["learning_rate"],
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        loss_fn = nn.L1Loss()  # MAE，与评估指标对齐

        # 6. 训练循环（含早停）
        best_val = float("inf")
        patience_left = self.params["patience"]
        best_state = None
        for epoch in range(1, self.params["epochs"] + 1):
            self.model_.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model_(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                # 梯度裁剪（防 LSTM 爆炸）
                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(),
                    max_norm=self.params["grad_clip"],
                )
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            train_mae = epoch_loss / max(n_batches, 1)

            # 验证（可选）
            val_mae = float("nan")
            if X_val is not None and y_val is not None:
                val_mae = self._compute_val_mae(X_val, y_val, window)

            logger.info(
                "LSTM epoch %d/%d  train_mae(log)=%.4f  val_mae(log)=%.4f  lr=%.2e",
                epoch, self.params["epochs"],
                train_mae, val_mae,
                optimizer.param_groups[0]["lr"],
            )

            scheduler.step(val_mae if not np.isnan(val_mae) else train_mae)

            # 早停判断（仅在有 val 时）
            if not np.isnan(val_mae):
                if val_mae < best_val - 1e-6:
                    best_val = val_mae
                    patience_left = self.params["patience"]
                    # 克隆到 CPU 避免 GPU 显存压力
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model_.state_dict().items()
                    }
                    # best.pt 落盘（防 SSH 断线丢进度）
                    if checkpoint_dir is not None:
                        self._save_checkpoint(
                            checkpoint_dir, "best.pt", best_state, best_val
                        )
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        logger.info("LSTM 早停于 epoch %d（val_mae 无改善）", epoch)
                        break

            # 每 N epoch 保存 latest.pt（断线后可手动 resume 或选最优）
            if checkpoint_dir is not None and epoch % self.params["save_every_epochs"] == 0:
                latest_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
                self._save_checkpoint(
                    checkpoint_dir, "latest.pt", latest_state, best_val
                )

        # 恢复 best_state（早停可能错过最优）
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        logger.info("LSTM 训练结束，best_val_mae(log)=%.4f", best_val)
        return self

    # ──────────── 推理 ────────────

    def _save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        filename: str,
        state_dict: dict,
        best_val: float,
    ) -> None:
        """训练过程中落盘（best.pt / latest.pt）。

        与 save() 区别：仅保存权重 + 标量，不依赖 self.model_ 当前状态，
        防止在加载 latest.pt 时与 best_val 不一致。
        """
        d = Path(checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": state_dict,
            "params": self.params,
            "feature_names": self.feature_names,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "best_val": float(best_val),
        }
        torch.save(ckpt, d / filename)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测。

        参数
        ----------
        X : DataFrame
            特征矩阵（n, n_features），与 train 时同列序。

        返回
        -------
        np.ndarray
            长度 n - window_size（每个窗口输出下一步预测）；原始单位（瓦）。
        """
        if self.model_ is None:
            raise RuntimeError("必须先调用 fit() 才能 predict")
        self._validate_xy(X)
        X_aligned = self._validate_feature_names(X)
        X_scaled = pd.DataFrame(
            self._scale(X_aligned),
            columns=X_aligned.columns,
            index=X_aligned.index,
        )
        window = self.params["window_size"]
        # 无 target 也可构造 SequenceDataset（target 传 dummy zeros）
        dummy_y = np.zeros(len(X_scaled), dtype=np.float32)
        loader = self._build_loader(
            X_scaled, dummy_y, window,
            batch_size=self.params["batch_size"] * 2,
            shuffle=False,
        )
        self.model_.eval()
        preds_log: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch, _ in loader:
                preds_log.append(self.model_(X_batch.to(self.device)).cpu().numpy())
        if not preds_log:
            return np.array([], dtype=np.float32)
        preds = np.concatenate(preds_log)
        return np.expm1(preds)  # 反变换回原始单位（瓦）

    # ──────────── 持久化 ────────────

    def save(self, path: Union[str, Path]) -> None:
        """torch.save 持久化。

        文件内容（dict）：
        - state_dict : _LSTMNet 权重
        - params : 架构超参（hidden_dim / num_layers / dropout / window_size / ...）
        - feature_names : 训练时特征列序
        - scaler_mean, scaler_std : 标准化器（numpy ndarray）

        git 已忽略 `*.pt`，不会污染仓库。
        """
        if self.model_ is None:
            raise RuntimeError("必须先调用 fit() 才能 save")
        if self._scaler_mean is None or self._scaler_std is None:
            raise RuntimeError("标准化器未初始化（fit 异常？）")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": self.model_.state_dict(),
            "params": self.params,
            "feature_names": self.feature_names,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
        }
        torch.save(ckpt, p)
        logger.info("LSTM checkpoint 已保存: %s", p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LSTMModel":
        """从 .pt 恢复 LSTMModel 实例。

        注意：torch.load 需 `weights_only=False` 以加载自定义 dict
        （PyTorch 2.6+ 默认 True，会拒绝自定义对象）。
        """
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(
            params=ckpt["params"],
            feature_names=ckpt["feature_names"],
        )
        # 恢复标准化器
        model._scaler_mean = ckpt["scaler_mean"]
        model._scaler_std = ckpt["scaler_std"]
        # 恢复 PyTorch 模型权重
        net = _LSTMNet(
            n_features=len(ckpt["feature_names"]),
            hidden_dim=ckpt["params"]["hidden_dim"],
            num_layers=ckpt["params"]["num_layers"],
            dropout=ckpt["params"]["dropout"],
        )
        net.load_state_dict(ckpt["state_dict"])
        net.to(model.device)
        model.model_ = net
        logger.info("LSTM checkpoint 已加载: %s", path)
        return model
