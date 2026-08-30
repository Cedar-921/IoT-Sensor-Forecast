"""src.sequence_dataset.SequenceDataset 的单元测试。

5 个测试，覆盖：长度、形状、target 对齐、不泄漏、防御异常。
合成 DataFrame，不依赖真实数据，秒级可跑（部署到云端后）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.sequence_dataset import SequenceDataset


def _make_synthetic(n: int = 100, n_features: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """合成特征 + target，便于测试窗口逻辑。"""
    rng = np.random.default_rng(seed=42)
    feat = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    target = pd.Series(rng.normal(size=n), name="Appliances")
    return feat, target


def test_dataset_known_length():
    """len(ds) 应等于 n - window_size。"""
    feat, target = _make_synthetic(n=100, n_features=5)
    window = 24
    ds = SequenceDataset(feat, target, window_size=window)
    assert len(ds) == 100 - window


def test_dataset_window_shape():
    """__getitem__(0) 应返回 (window_size, n_features) 张量 + 标量。"""
    feat, target = _make_synthetic(n=100, n_features=5)
    window = 24
    ds = SequenceDataset(feat, target, window_size=window)
    X, y = ds[0]

    # X 是 (window_size, n_features) float32
    assert isinstance(X, torch.Tensor)
    assert X.shape == (window, 5)
    assert X.dtype == torch.float32

    # y 是标量 float32
    assert isinstance(y, torch.Tensor)
    assert y.ndim == 0  # 0-dim 标量张量
    assert y.dtype == torch.float32


def test_dataset_target_alignment():
    """第 i 个样本的 y 应等于原始 target.iloc[i + window_size]。"""
    feat, target = _make_synthetic(n=100, n_features=3)
    window = 10
    ds = SequenceDataset(feat, target, window_size=window)

    for i in [0, 5, 50, 80]:
        X, y = ds[i]
        # target.iloc[i + window_size] 对应窗口末尾的下一时刻
        expected_y = float(target.iloc[i + window_size])
        assert y.item() == pytest.approx(expected_y, rel=1e-6)

        # X 应等于 feat.iloc[i:i+window].values
        expected_X = feat.iloc[i : i + window].values.astype(np.float32)
        np.testing.assert_allclose(X.numpy(), expected_X, rtol=1e-6)


def test_dataset_no_leakage():
    """窗口内时间步必须严格早于预测时刻（防止未来信息泄漏）。"""
    feat, target = _make_synthetic(n=50, n_features=3)
    window = 10
    ds = SequenceDataset(feat, target, window_size=window)

    # 取第 i 个样本，X 来自 [i, i+window)，y 来自 i+window
    # 验证 X 中**不包含** y 对应的特征值
    i = 20
    X, y = ds[i]
    # 第 i+window 时刻的特征（即 y 对应行的特征）不应在 X 内
    forbidden = feat.iloc[i + window].values
    X_np = X.numpy()
    # forbidden 的每一列都不应在 X 的对应列范围 [i, i+window) 出现
    # 用精确匹配确认（合成数据，值唯一）
    for col_idx in range(X_np.shape[1]):
        assert forbidden[col_idx] not in X_np[:, col_idx], (
            f"窗口泄漏：第 {i+window} 时刻的特征出现在第 {i} 个样本的窗口中"
        )


def test_dataset_empty_or_short_raises():
    """长度不足或不一致应抛 ValueError。"""
    feat, target = _make_synthetic(n=100, n_features=5)

    # 样本数 < window_size + 1
    with pytest.raises(ValueError, match="样本数"):
        SequenceDataset(feat.iloc[:10], target.iloc[:10], window_size=24)

    # features/target 长度不一致
    with pytest.raises(ValueError, match="长度不一致"):
        SequenceDataset(feat.iloc[:50], target.iloc[:30], window_size=10)


def test_dataset_resets_non_standard_index():
    """入参索引非 0-based 时应 reset_index 后正确取窗口。"""
    feat, target = _make_synthetic(n=100, n_features=3)
    # 给一个非默认索引
    feat.index = pd.date_range("2020-01-01", periods=100, freq="10min")
    target.index = feat.index

    window = 24
    ds = SequenceDataset(feat, target, window_size=window)

    # 即使原索引是 DatetimeIndex，ds[0] 仍应正确取前 24 行
    X, y = ds[0]
    assert X.shape == (window, 3)
    # y 应等于第 window 行的 target（reset_index 后第 window 行）
    assert y.item() == pytest.approx(float(target.iloc[window]), rel=1e-6)
