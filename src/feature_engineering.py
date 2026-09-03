"""IoT-Sensor-Forecast · 特征工程模块。

提供两个公开函数：
- fit_transform_features(train_df)：在训练集上构造特征，返回 (train_feat, feature_cols, train_tail)。
- transform_features(df, train_tail)：把 train_tail 拼接到 df 前面，再计算 lag/rolling，
  切片返回 df 部分，避免 val/test 首行 lag/rolling 为 NaN。

设计要点：
- 周期编码（sin/cos）替代原始 hour/dayofweek/month。
- rolling 前 shift(1)：保证 rolling 不包含当前时刻（防泄漏）。
- lag/rolling 计算时使用全部可用历史；首部 NaN 由调用方 dropna。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────── 特征配置常量（便于测试 monkeypatch） ───────────────
LAG_PERIODS: tuple[int, ...] = (1, 3, 6, 12, 24, 288)
ROLLING_WINDOWS: tuple[int, ...] = (144,)
ROLLING_STATS: tuple[str, ...] = ("mean", "std")
TARGET_COL: str = "Appliances"
TARGET_LOG_COL: str = "Appliances_log"

# 室内温度（T6 已清洗删除）
INDOOR_TEMP_COLS: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9")
INDOOR_HUM_COLS: tuple[str, ...] = (
    "RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9",
)
OUTDOOR_COLS: tuple[str, ...] = (
    "T_out", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint",
)
OTHER_COLS: tuple[str, ...] = ("lights", "rv1")

# 原始特征列全集（与建模列并列，作为底座保留）
RAW_FEATURE_COLS: frozenset[str] = frozenset([
    *INDOOR_TEMP_COLS, *INDOOR_HUM_COLS, *OUTDOOR_COLS, *OTHER_COLS,
])

# 时间编码列
TIME_COLS: frozenset[str] = frozenset({
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",
})

# ─────────────── 周期编码辅助 ───────────────
_PERIODS: dict[str, int] = {
    "hour": 24,
    "dow": 7,
    "month": 12,
}


def _cyclic_encode(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """对一组整数值（小时/星期/月）做 sin/cos 周期编码。"""
    radians = 2.0 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 datetime 索引衍生时间特征。"""
    out = df.copy()
    idx = out.index

    hour = idx.hour
    dow = idx.dayofweek
    month = idx.month

    out["hour_sin"], out["hour_cos"] = _cyclic_encode(hour, _PERIODS["hour"])
    out["dow_sin"], out["dow_cos"] = _cyclic_encode(dow, _PERIODS["dow"])
    # month 为 1-indexed（January=1, December=12），用 (month-1)/12 使 January 位于 0°
    # 与 hour/dow 的 0-indexed 编码逻辑一致，避免 month=12 和 month=0 落在同一位
    out["month_sin"], out["month_cos"] = _cyclic_encode(month - 1, _PERIODS["month"])

    out["is_weekend"] = (dow >= 5).astype(int)
    return out


def _add_lag_rolling(
    combined: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """对 combined（含历史 + 目标期间）追加 lag/rolling 特征。

    假设 combined 已经按时间排序；最终返回切片只保留末尾 len(target_len) 行。
    """
    out = combined.copy()
    series = out[target_col].astype(float)

    # lag 特征（不泄漏：用 shift(k) 表示 t-k 时刻的目标）
    for k in LAG_PERIODS:
        out[f"{target_col}_lag{k}"] = series.shift(k)

    # rolling 特征（防泄漏：先 shift(1) 再 rolling）
    shifted = series.shift(1)
    for w in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            col_name = f"{target_col}_roll{stat}_{w}"
            if stat == "mean":
                out[col_name] = shifted.rolling(window=w, min_periods=w).mean()
            elif stat == "std":
                out[col_name] = shifted.rolling(window=w, min_periods=w).std()
    return out


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """在已有 DataFrame 上追加时间特征 + lag/rolling 特征。"""
    out = _add_time_features(df)
    out = _add_lag_rolling(out, TARGET_COL)
    return out


def _resolve_feature_cols(df_with_features: pd.DataFrame) -> list[str]:
    """从含特征的 DataFrame 中选出建模列。

    保留策略：原始特征列（室内/室外/其他）+ 时间编码列 + 衍生特征列。
    新增衍生特征（lag/roll/ewm 结尾）自动纳入，无需改这里。
    """
    exclude = {TARGET_COL, TARGET_LOG_COL}
    keep = []
    for c in df_with_features.columns:
        if c in exclude:
            continue
        # 原始特征列（室内温度/湿度、室外、其他）
        if c in RAW_FEATURE_COLS:
            keep.append(c)
        # 时间编码列
        elif c in TIME_COLS:
            keep.append(c)
        # 衍生特征：以目标列名开头（Appliances_lag* / Appliances_roll* / Appliances_ewm*）
        elif c.startswith((f"{TARGET_COL}_lag", f"{TARGET_COL}_roll", f"{TARGET_COL}_ewm")):
            keep.append(c)
    return keep


def _calc_warmup_size() -> int:
    """计算需要的 warmup 行数 = max(最大 lag, 最大 rolling 窗口)。"""
    return max(max(LAG_PERIODS, default=0), max(ROLLING_WINDOWS, default=0))


# ─────────────── 公开 API ───────────────
def fit_transform_features(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """对训练集构造特征。

    参数
    ----------
    train_df : pd.DataFrame
        训练集原始数据（要求 index 是 datetime 类型）。

    返回
    -------
    train_feat : pd.DataFrame
        训练特征矩阵。包含：
          - 原始列（保留）
          - 衍生时间/周期/星期特征
          - lag/rolling 特征（首部会有 NaN，由调用方 dropna）
          - Appliances_log 目标列（保留以便训练）
    feature_cols : list[str]
        建模用的特征列名（不含目标列、不含原始冗余列）。
    train_tail : pd.DataFrame
        训练集末尾 warmup 行（行数 = max(最大 lag, 最大 rolling 窗口)）。
        供后续 transform_features 拼接在 val/test 前面，避免 val/test 首行 NaN。
    """
    df = train_df.copy(deep=True)
    warmup = _calc_warmup_size()
    train_tail = df.iloc[-warmup:].copy() if warmup > 0 else df.iloc[0:0].copy()

    feat = _build_feature_frame(df)

    # 确保 Appliances_log 存在（若上游未提供则补算）
    if TARGET_LOG_COL not in feat.columns and TARGET_COL in feat.columns:
        feat[TARGET_LOG_COL] = np.log1p(feat[TARGET_COL].astype(float))

    feature_cols = _resolve_feature_cols(feat)
    return feat, feature_cols, train_tail


def transform_features(
    df: pd.DataFrame,
    train_tail: pd.DataFrame,
) -> pd.DataFrame:
    """对任意 DataFrame（val / test）构造特征。

    把 train_tail 拼接到 df 前面计算 lag/rolling，再切片返回 df 部分，
    保证 df 首行的 lag/rolling 不为 NaN（除非 warmup 不足，这种情况会留 NaN 给调用方处理）。

    参数
    ----------
    df : pd.DataFrame
        待构造特征的 DataFrame（val 或 test）。
    train_tail : pd.DataFrame
        fit_transform_features 返回的 train_tail。
    """
    df_in = df.copy(deep=True)
    # 用 train_tail 提供历史暖机；拼接计算后切片返回目标部分
    if len(train_tail) > 0:
        combined = pd.concat([train_tail, df_in], axis=0)
    else:
        combined = df_in

    # combined 必须保留 df_in 的所有原始列，所以保留整个 df_in 的列
    feat_full = _build_feature_frame(combined)

    # 切片：保留 df_in 的行
    n = len(df_in)
    feat = feat_full.iloc[-n:].copy()

    # 补 Appliances_log（val/test 通常仍保留 Appliances 原列）
    if TARGET_LOG_COL not in feat.columns and TARGET_COL in feat.columns:
        feat[TARGET_LOG_COL] = np.log1p(feat[TARGET_COL].astype(float))

    return feat


def drop_initial_nans(
    df: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """丢掉任意 feature 列为 NaN 的行（lag/rolling 引入）。"""
    before = len(df)
    out = df.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info("drop_initial_nans: %d → %d 行", before, len(out))
    return out