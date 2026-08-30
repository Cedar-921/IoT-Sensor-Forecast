"""src.data_cleaning 模块的单元测试。

覆盖 6 个用例：列删除、日期解析、log 变换、端到端管道、
入参不可变性、CSV 读写往返。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    clean_data,
    load_raw_data,
    run_pipeline,
    save_cleaned_data,
)


# ───────────────────────── Fixtures ─────────────────────────
@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """模块级别 fixture：复用真实原始数据，所有测试共享。"""
    return load_raw_data()


@pytest.fixture
def cleaned(raw_df: pd.DataFrame) -> pd.DataFrame:
    """函数级别 fixture：每个测试独立的清洗结果。"""
    return clean_data(raw_df)


# ───────────────────────── 测试用例 ─────────────────────────
def test_drop_columns(cleaned: pd.DataFrame) -> None:
    """验证 T6 和 rv2 已删除，关键列保留。"""
    assert "T6" not in cleaned.columns, "T6 应被删除"
    assert "rv2" not in cleaned.columns, "rv2 应被删除"
    expected_kept = {"T1", "RH_1", "Appliances", "Appliances_log", "date"}
    missing = expected_kept - set(cleaned.columns)
    assert not missing, f"关键列缺失：{missing}"
    # 原始 29 列 - 2 列删除 + 1 列新增 = 28 列
    assert cleaned.shape[1] == 28, f"期望 28 列，实际 {cleaned.shape[1]}"


def test_fix_date(cleaned: pd.DataFrame) -> None:
    """验证 date 列被正确解析为 datetime64 类型，且无 NaT。"""
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"]), (
        f"date 必须是 datetime64 类型，实际是 {cleaned['date'].dtype}"
    )
    assert cleaned["date"].isna().sum() == 0, "date 不应有 NaT"
    # 首行应解析为 2016-01-11 17:00:00（验证正则修复了缺空格 bug）
    assert cleaned["date"].iloc[0] == pd.Timestamp("2016-01-11 17:00:00"), (
        f"首行日期解析错误：{cleaned['date'].iloc[0]}"
    )
    # 时间范围应跨多个月
    assert (cleaned["date"].max() - cleaned["date"].min()).days > 100, (
        "时间跨度应超过 100 天"
    )


def test_log_transform(cleaned: pd.DataFrame) -> None:
    """验证 Appliances_log 列范围和首行值正确。"""
    s = cleaned["Appliances_log"]
    # log1p(10) ≈ 2.3979, log1p(1080) ≈ 6.9856
    assert s.min() >= np.log1p(10) - 1e-6, f"log 最小值偏小：{s.min()}"
    assert s.max() <= np.log1p(1080) + 1e-6, f"log 最大值偏大：{s.max()}"
    # 首行 Appliances=60 → log1p(60) ≈ 4.1109
    np.testing.assert_allclose(
        s.iloc[0], np.log1p(60), rtol=1e-6,
        err_msg=f"首行 log 值错误：{s.iloc[0]} ≠ log1p(60)={np.log1p(60)}"
    )
    # 反变换无损
    np.testing.assert_allclose(
        np.expm1(s.values), cleaned["Appliances"].values, rtol=1e-6
    )


def test_pipeline_runs(tmp_path, monkeypatch) -> None:
    """端到端：monkeypatch 输出路径到 tmp_path，验证管道成功执行。"""
    out_path = tmp_path / "cleaned.csv"
    monkeypatch.setattr("src.data_cleaning.CLEAN_PATH", out_path)
    df = run_pipeline()
    assert out_path.exists(), f"管道未生成 CSV：{out_path}"
    assert df.shape == (19735, 28), f"shape 错误：{df.shape}"


def test_no_modify_input(raw_df: pd.DataFrame) -> None:
    """验证 clean_data 不修改入参（防止 df 是视图的坑）。"""
    snapshot = raw_df.copy(deep=True)
    _ = clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, snapshot, check_exact=False, rtol=1e-6, atol=1e-9)


def test_saves_csv(tmp_path, cleaned: pd.DataFrame) -> None:
    """验证 save_cleaned_data 写 CSV 后可读回，且 dtype 保留。"""
    out_path = tmp_path / "out.csv"
    save_cleaned_data(cleaned, out_path)
    assert out_path.exists()
    back = pd.read_csv(out_path, parse_dates=["date"])
    assert back.shape == cleaned.shape
    assert pd.api.types.is_datetime64_any_dtype(back["date"])
    # 列顺序保持一致
    assert list(back.columns) == list(cleaned.columns)


def test_clean_data_drops_nat_rows(raw_df: pd.DataFrame) -> None:
    """Step 3 兜底：date 含 NaT 应被 dropna。"""
    df = raw_df.copy()
    # 在 date 列混入几条非法字符串，制造 NaT
    df.loc[df.index[10], "date"] = "not-a-date"
    df.loc[df.index[100], "date"] = "garbage"
    n_before = len(df)
    cleaned = clean_data(df)
    # NaT 行被丢掉
    assert len(cleaned) == n_before - 2
    # 没有 NaT 残留
    assert cleaned["date"].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])