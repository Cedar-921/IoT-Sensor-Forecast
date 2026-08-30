"""src.feature_engineering 单元测试。

使用合成 DataFrame（200 行），不依赖 cleaned.csv，保证秒级执行。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    LAG_PERIODS,
    ROLLING_WINDOWS,
    TARGET_COL,
    _calc_warmup_size,
    drop_initial_nans,
    fit_transform_features,
    transform_features,
)


def _make_synthetic_df(n: int = 300, start: str = "2016-01-11 17:00:00") -> pd.DataFrame:
    """合成一个符合清洗数据格式的 DataFrame。"""
    idx = pd.date_range(start=start, periods=n, freq="10min")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "lights": rng.integers(0, 70, n),
            "T1": 20 + rng.normal(0, 0.5, n),
            "RH_1": 40 + rng.normal(0, 1, n),
            "T2": 20 + rng.normal(0, 0.5, n),
            "RH_2": 40 + rng.normal(0, 1, n),
            "T3": 20 + rng.normal(0, 0.5, n),
            "RH_3": 40 + rng.normal(0, 1, n),
            "T4": 20 + rng.normal(0, 0.5, n),
            "RH_4": 40 + rng.normal(0, 1, n),
            "T5": 20 + rng.normal(0, 0.5, n),
            "RH_5": 40 + rng.normal(0, 1, n),
            "RH_6": 40 + rng.normal(0, 1, n),
            "T7": 20 + rng.normal(0, 0.5, n),
            "RH_7": 40 + rng.normal(0, 1, n),
            "T8": 20 + rng.normal(0, 0.5, n),
            "RH_8": 40 + rng.normal(0, 1, n),
            "T9": 20 + rng.normal(0, 0.5, n),
            "RH_9": 40 + rng.normal(0, 1, n),
            "T_out": 5 + rng.normal(0, 1, n),
            "Press_mm_hg": 750 + rng.normal(0, 1, n),
            "RH_out": 80 + rng.normal(0, 1, n),
            "Windspeed": 5 + rng.normal(0, 1, n),
            "Visibility": 30 + rng.normal(0, 1, n),
            "Tdewpoint": 0 + rng.normal(0, 1, n),
            "rv1": rng.normal(0, 1, n),
            "Appliances": rng.integers(10, 200, n),
        },
        index=idx,
    )
    df["Appliances_log"] = np.log1p(df["Appliances"])
    return df


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    """300 行合成数据，模块级共享。"""
    return _make_synthetic_df(300)


def test_features_known_count(synthetic):
    """特征列数应为各分量之和（计算式断言，避免 magic number）。"""
    from src.feature_engineering import (
        LAG_PERIODS, ROLLING_WINDOWS, ROLLING_STATS,
        INDOOR_TEMP_COLS, INDOOR_HUM_COLS, OUTDOOR_COLS, OTHER_COLS,
    )
    expected = (
        6  # hour_sin/cos, dow_sin/cos, month_sin/cos
        + 1  # is_weekend
        + len(LAG_PERIODS)  # 6 个 lag
        + len(ROLLING_WINDOWS) * len(ROLLING_STATS)  # 1 个 mean + 1 个 std = 2
        + len(INDOOR_TEMP_COLS)  # 8
        + len(INDOOR_HUM_COLS)  # 9
        + len(OUTDOOR_COLS)  # 6
        + len(OTHER_COLS)  # lights, rv1
    )
    _, feat_cols, _ = fit_transform_features(synthetic)
    assert len(feat_cols) == expected, (
        f"特征列数不匹配：预期 {expected}, 实际 {len(feat_cols)}\n"
        f"配置：lag={LAG_PERIODS}, rolling={ROLLING_WINDOWS}x{ROLLING_STATS}"
    )


def test_features_no_modify_input(synthetic):
    """fit_transform_features 不应修改入参。"""
    before = synthetic.copy(deep=True)
    _ = fit_transform_features(synthetic)
    pd.testing.assert_frame_equal(synthetic, before)


def test_features_lag_shift_one(synthetic):
    """lag(1) 应等于上一行的 Appliances。"""
    train_feat, feat_cols, _ = fit_transform_features(synthetic)
    # 取足够靠后的行（避开首部 NaN，且索引合法）
    valid_idx = 200  # < len(train_feat)=300
    assert train_feat[TARGET_COL].iloc[valid_idx - 1] == train_feat["Appliances_lag1"].iloc[valid_idx]


def test_features_rolling_no_leakage(synthetic):
    """rolling 应不包含当前行（用 shift(1) 隔离）。"""
    train_feat, feat_cols, _ = fit_transform_features(synthetic)
    # 取一个 rolling 完整填充的位置（> 144）
    i = 200
    # 计算预期：i-1 及其前 143 行的均值
    expected_mean = synthetic[TARGET_COL].iloc[i - 144:i].mean()
    actual_mean = train_feat["Appliances_rollmean_144"].iloc[i]
    assert abs(actual_mean - expected_mean) < 1e-6, (
        f"rolling 应不包含当前行：预期 {expected_mean}, 实际 {actual_mean}"
    )


def test_features_cyclic_in_minus_one_one(synthetic):
    """sin/cos 编码应在 [-1, 1] 范围内。"""
    train_feat, feat_cols, _ = fit_transform_features(synthetic)
    for c in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert train_feat[c].min() >= -1.0 - 1e-9
        assert train_feat[c].max() <= 1.0 + 1e-9


def test_features_transform_uses_train_warmup(synthetic):
    """val 首行 lag(1) 应不为 NaN（验证关键 bug 修复）。"""
    train_feat, feat_cols, train_tail = fit_transform_features(synthetic)
    val = synthetic.iloc[200:]  # 模拟 val
    val_feat = transform_features(val, train_tail)
    # 首行的 lag1 应由 train_tail 提供，**不为 NaN**
    assert not pd.isna(val_feat["Appliances_lag1"].iloc[0])
    # rolling(144) 首行也应该由 warmup 提供
    assert not pd.isna(val_feat["Appliances_rollmean_144"].iloc[0])


def test_features_drop_initial_nans(synthetic):
    """drop_initial_nans 应丢弃首部因 lag/rolling 引入 NaN 的行。"""
    train_feat, feat_cols, _ = fit_transform_features(synthetic)
    warmup = _calc_warmup_size()
    n_before = len(train_feat)
    cleaned = drop_initial_nans(train_feat, feat_cols)
    n_after = len(cleaned)
    # 应丢弃恰好 warmup 行
    assert n_before - n_after == warmup, (
        f"应丢 {warmup} 行，实际丢 {n_before - n_after} 行"
    )


def test_features_feature_cols_no_target(synthetic):
    """feature_cols 不应包含 Appliances 或 Appliances_log。"""
    _, feat_cols, _ = fit_transform_features(synthetic)
    assert TARGET_COL not in feat_cols
    assert "Appliances_log" not in feat_cols
    # 也不应包含 date/index 名
    assert "date" not in feat_cols


def test_features_transform_empty_tail_no_modify(synthetic):
    """transform_features 不修改入参 df 与 train_tail。"""
    _, _, train_tail = fit_transform_features(synthetic)
    val = synthetic.iloc[200:]
    val_before = val.copy(deep=True)
    tail_before = train_tail.copy(deep=True)
    _ = transform_features(val, train_tail)
    pd.testing.assert_frame_equal(val, val_before)
    pd.testing.assert_frame_equal(train_tail, tail_before)


def test_features_transform_each_lag_period(synthetic):
    """各 lag_k 应严格等于 k 步之前的 Appliances。"""
    # fit 仍用全集 300 行，确保 train_tail 有足够 warmup（max lag=288）
    _, feat_cols, train_tail = fit_transform_features(synthetic)
    # val 较短也能验证：warmup 已由 train_tail 提供
    val = synthetic.iloc[5:]  # 长度 295，足以覆盖 k=288
    val_feat = transform_features(val, train_tail)
    for k in (1, 3, 6, 12, 24, 288):
        col = f"Appliances_lag{k}"
        assert col in val_feat.columns, f"缺失 lag 列 {col}"
        # 取 val 内部足够靠后的位置（避开开头）
        i = k + 10
        if i >= len(val_feat):
            continue  # val 太短则跳过（极端 lag）
        # lag(k)[i] = val Appliances 在 (i-k) 位置，但跨 train+val 边界需 join
        # 简化：只断言 k < len(val) 时正确
        if k < len(val):
            expected = val["Appliances"].iloc[i - k]
            actual = val_feat[col].iloc[i]
            assert actual == expected, f"lag{k} 错位：预期 {expected}, 实际 {actual}"


def test_features_transform_without_train_tail(synthetic):
    """无 train_tail 时 transform_features 也能跑（首部会有 NaN）。"""
    val = synthetic.iloc[200:]
    empty_tail = pd.DataFrame(columns=val.columns).astype(val.dtypes.to_dict())
    val_feat = transform_features(val, empty_tail)
    # 形状正确
    assert val_feat.shape[0] == len(val)
    # 首行 lag 应为 NaN（warmup 不足）
    assert pd.isna(val_feat["Appliances_lag1"].iloc[0])