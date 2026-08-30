"""IoT-Sensor-Forecast · 滑动窗口 Dataset。

把 (n_samples, n_features) 的特征矩阵转成
(n_samples - window_size, window_size, n_features) 的窗口序列，
每条样本的 target 是窗口末尾的下一时刻。

为什么不自己用 list 实现 __getitem__：
- torch DataLoader 自动处理 batch / shuffle / pin_memory
- 与 PyTorch 生态对齐（多 worker、GPU 加速）
- 后续 Transformer 复用同一套 Dataset，零成本迁移
"""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """LSTM 滑动窗口数据集。"""

    def __init__(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        window_size: int = 24,
    ) -> None:
        """构造滑动窗口数据集。

        参数
        ----------
        features : pd.DataFrame
            (n, n_features)，**已标准化**（LSTMModel 内部 fit 时做）。
        target : pd.Series
            (n,)，原始单位 Appliances（log1p 在 LSTMModel 内部做）。
        window_size : int
            窗口长度（多少个时间步预测下一步）。

        异常
        ------
        ValueError
            features/target 长度不一致或样本数不足以构造一个完整窗口。
        """
        if len(features) != len(target):
            raise ValueError(
                f"features/target 长度不一致：{len(features)} vs {len(target)}"
            )
        if len(features) < window_size + 1:
            raise ValueError(
                f"样本数 {len(features)} 不足 window_size+1={window_size + 1}"
            )
        # reset_index 防止入参索引非 0-based 时 iloc 错位
        self.features = features.reset_index(drop=True)
        self.target = target.reset_index(drop=True)
        self.window_size = window_size

    def __len__(self) -> int:
        """可构造的窗口数 = 总样本数 - 窗口大小。

        举例：1000 行数据，window=24 → 976 个样本（每个对应一个 [i, i+24) 窗口 → y=i+24）。
        """
        return len(self.features) - self.window_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (window, target)：
        - X : (window_size, n_features) float32
        - y : 标量 float32（窗口末尾的下一时刻 Appliances）

        窗口右开：`[idx, idx+window_size)` → target 是 `idx+window_size`。
        """
        X = self.features.iloc[idx : idx + self.window_size].values
        y = self.target.iloc[idx + self.window_size]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(float(y), dtype=torch.float32),
        )
