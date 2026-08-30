"""IoT-Sensor-Forecast · Transformer Encoder 时序预测模型。

API 镜像 src.models.lstm_model.LSTMModel：
- fit(X, y) / fit(X, y, X_val, y_val)
- predict(X) -> np.ndarray（原始单位）
- save(path) / load(path) classmethod

内部细节（调用方不感知）：
- log1p(y) / expm1(pred) 与 LSTM/XGBoost 对齐
- 标准正弦位置编码（不可学习，业界标准）
- TransformerEncoderLayer(d_model=hidden_dim, nhead, dim_feedforward, dropout)
- device 自动选 cuda/cpu
- 早停 + ReduceLROnPlateau

云端训练约定（详见 CLAUDE.md「训练环境（云端 GPU）」）：
- 云服务器 pip install torch --index-url .../whl/cu118（按 CUDA 版本）
- nohup python src/train.py --model transformer > logs/transformer.log 2>&1 &
- checkpoint 保存到 models/transformer_v1.pt（git 已忽略）
"""
from __future__ import annotations

import logging
import math
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

class _PositionalEncoding(nn.Module):
    """标准正弦位置编码，不可学习。

    公式：PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
         PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    为什么不用可学习的位置嵌入：
    - 时序数据位置是严格有序的，正弦编码归纳偏置更强
    - 可学习嵌入对短序列容易过拟合
    - 与原始 Transformer 论文（Vaswani et al., 2017）一致
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) 方便 broadcast
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class _TransformerNet(nn.Module):
    """Transformer Encoder 时序预测模型。

    结构：Input(batch, window_size, n_features)
        → Linear(n_features → hidden_dim)   # 输入投影
        → PositionalEncoding(hidden_dim)
        → TransformerEncoder(num_layers, nhead, dim_feedforward, dropout)
        → 取最后一个时间步 → Linear(hidden_dim → 1) → squeeze

    为什么取最后一个时间步而非 CLS token：
    - 时序预测不需要 [CLS] 的全局语义，末尾位置已聚合全部历史信息
    - 避免引入额外可学习 [CLS] 向量，简化实现
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # 输入投影：n_features → hidden_dim
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.pos_encoder = _PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # 与 LSTM 一样用 batch_first
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, n_features)
        x = self.input_proj(x)  # (batch, window, hidden_dim)
        x = self.pos_encoder(x)  # (batch, window, hidden_dim)
        out = self.transformer(x)  # (batch, window, hidden_dim)
        last = out[:, -1, :]  # (batch, hidden_dim) 末尾时间步
        return self.head(last).squeeze(-1)  # (batch,)


# ─────────────── 公开 TransformerModel ───────────────

class TransformerModel:
    """Transformer 时序预测模型（包装类，调用方不接触 torch 细节）。

    API 与 LSTMModel 100% 对齐。
    """

    DEFAULT_PARAMS: dict = {
        "window_size": 24,
        "hidden_dim": 64,
        "nhead": 4,  # 必须整除 hidden_dim
        "num_layers": 2,
        "dim_feedforward": 128,  # 标准：hidden_dim * 2
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "epochs": 30,
        "patience": 5,
        "grad_clip": 1.0,
        "random_state": 42,
        "save_every_epochs": 5,
    }

    def __init__(
        self,
        params: dict | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """构造 TransformerModel。

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
        self.model_: _TransformerNet | None = None
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
            num_workers=0,  # Windows 安全；云端 Linux 可改 2-4
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
    ) -> "TransformerModel":
        """训练 Transformer。

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
            None = 不保存。

        返回
        -------
        TransformerModel
            self（链式调用）。
        """
        self._validate_xy(X, y)
        self._validate_no_nan(X, y)

        # 1. 标准化
        self._scaler_mean = X.mean(axis=0).values.astype(np.float32)
        std = X.std(axis=0).values.astype(np.float32)
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
        self.model_ = _TransformerNet(
            n_features=X_scaled.shape[1],
            hidden_dim=self.params["hidden_dim"],
            nhead=self.params["nhead"],
            num_layers=self.params["num_layers"],
            dim_feedforward=self.params["dim_feedforward"],
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
                # 梯度裁剪（防 Transformer 训练不稳定）
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
                "Transformer epoch %d/%d  train_mae(log)=%.4f  val_mae(log)=%.4f  lr=%.2e",
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
                        logger.info("Transformer 早停于 epoch %d（val_mae 无改善）", epoch)
                        break

            # 每 N epoch 保存 latest.pt
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

        logger.info("Transformer 训练结束，best_val_mae(log)=%.4f", best_val)
        return self

    # ──────────── 推理 ────────────

    def _save_checkpoint(
        self,
        checkpoint_dir: str | Path,
        filename: str,
        state_dict: dict,
        best_val: float,
    ) -> None:
        """训练过程中落盘（best.pt / latest.pt）。"""
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
        - state_dict : _TransformerNet 权重
        - params : 架构超参（hidden_dim / nhead / num_layers / dim_feedforward / dropout / window_size / ...）
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
        logger.info("Transformer checkpoint 已保存: %s", p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransformerModel":
        """从 .pt 恢复 TransformerModel 实例。

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
        net = _TransformerNet(
            n_features=len(ckpt["feature_names"]),
            hidden_dim=ckpt["params"]["hidden_dim"],
            nhead=ckpt["params"]["nhead"],
            num_layers=ckpt["params"]["num_layers"],
            dim_feedforward=ckpt["params"]["dim_feedforward"],
            dropout=ckpt["params"]["dropout"],
        )
        net.load_state_dict(ckpt["state_dict"])
        net.to(model.device)
        model.model_ = net
        logger.info("Transformer checkpoint 已加载: %s", path)
        return model
