"""IoT-Sensor-Forecast · 数据清洗模块

清洗 UCI 374 Appliances Energy 数据集中的 5 个质量问题（来自
`notebooks/01_eda.ipynb` cell-12 的关键发现）：

1. 删除冗余列 ``rv2``（与 ``rv1`` 完全相同，相关系数=1.0）
2. 删除异常传感器 ``T6``（与室外温度 ``T_out`` 相关 0.975，
   疑似室外探头被误标为室内房间）
3. 修复 ``date`` 列字符串拼接 bug（原始 ``"2016-01-1117:00:00"``
   缺分隔空格），正则插入空格后 ``pd.to_datetime`` 解析
4. 验证 ``date`` 全部解析成功（无 NaT），有异常则日志 warning 并 dropna
5. 对目标变量 ``Appliances`` 做 ``log1p`` 变换，新增
   ``Appliances_log`` 列；保留原列便于反变换与 EDA 对比

清洗结果保存到 ``data/processed/cleaned.csv``（被 ``.gitignore`` 排除）。

典型用法：

>>> from src.data_cleaning import run_pipeline
>>> df = run_pipeline()
>>> df.shape
(19735, 28)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# ───────────────────────── 路径常量（便于测试 monkeypatch） ─────────────────────────
RAW_PATH = Path("data/raw/appliances_energy.csv")
CLEAN_PATH = Path("data/processed/cleaned.csv")

# ───────────────────────── 清洗规则常量 ─────────────────────────
DROP_COLS: list[str] = ["rv2", "T6"]
DATE_REGEX: str = r"(\d{4}-\d{2}-\d{2})(\d{2}:)"
DATE_REPL: str = r"\1 \2"
LOG_TARGET: str = "Appliances_log"

# ───────────────────────── 日志 ─────────────────────────
logger = logging.getLogger(__name__)


def load_raw_data(path: Union[str, Path] = RAW_PATH) -> pd.DataFrame:
    """从指定路径读取原始 CSV。

    参数
    ----------
    path : str | Path
        默认 ``RAW_PATH``，测试可通过 monkeypatch 替换。

    返回
    -------
    pd.DataFrame
        原始数据，未做任何修改。

    异常
    ------
    FileNotFoundError
        当路径不存在时抛出。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"原始 CSV 不存在：{p}")
    df = pd.read_csv(p)
    logger.info("已加载原始数据 shape=%s, path=%s", df.shape, p)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """执行 4 步清洗流程，返回新的 DataFrame（不修改入参）。

    步骤
    ----
    1. 删除冗余/异常列（``rv2``、``T6``），列不存在时跳过（幂等）
    2. 修复 ``date`` 字符串拼接 bug 并 ``pd.to_datetime`` 解析
    3. 验证 ``date`` 无 NaT；如有则日志 warning 并 dropna
    4. 新增 ``Appliances_log = log1p(Appliances)``，保留原列

    参数
    ----------
    df : pd.DataFrame
        原始数据，应包含 ``rv2`` / ``T6`` / ``date`` / ``Appliances`` 列。

    返回
    -------
    pd.DataFrame
        清洗后数据，shape 从 ``(N, 29)`` 变为 ``(N, 28)``（删 2 列 + 加 1 列）。
    """
    out = df.copy(deep=True)  # 防御：防切片视图原地修改污染入参

    # ───── Step 1: 删冗余/异常列（幂等） ─────
    existing_drops = [c for c in DROP_COLS if c in out.columns]
    if existing_drops:
        out = out.drop(columns=existing_drops)
        logger.info("Step 1: 已删除列 %s，剩余列数=%d", existing_drops, out.shape[1])

    # ───── Step 2: 修复 date 字符串拼接 bug ─────
    out["date"] = out["date"].astype(str).str.replace(DATE_REGEX, DATE_REPL, regex=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # ───── Step 3: 验证日期无 NaT，失败则日志 warning 并 dropna ─────
    nat_n = int(out["date"].isna().sum())
    if nat_n > 0:
        logger.warning("date 解析产生 %d 行 NaT，已 dropna", nat_n)
        out = out.dropna(subset=["date"]).reset_index(drop=True)
    else:
        logger.info("Step 3: date 全部解析成功，dtype=%s", out["date"].dtype)

    # ───── Step 4: 目标对数变换 ─────
    out[LOG_TARGET] = np.log1p(out["Appliances"])
    logger.info(
        "Step 4: 已生成 %s 列，min=%.4f, max=%.4f",
        LOG_TARGET,
        out[LOG_TARGET].min(),
        out[LOG_TARGET].max(),
    )

    return out


def save_cleaned_data(
    df: pd.DataFrame, path: Union[str, Path] = CLEAN_PATH
) -> None:
    """将清洗结果写入 CSV，自动创建父目录。

    参数
    ----------
    df : pd.DataFrame
        ``clean_data`` 的输出。
    path : str | Path
        默认 ``CLEAN_PATH``，测试可通过 monkeypatch 替换到 tmp_path。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info("已保存清洗数据 shape=%s -> %s", df.shape, p)


def run_pipeline() -> pd.DataFrame:
    """端到端入口：load → clean → save → return。

    注意：显式读取模块级常量 ``CLEAN_PATH`` 后再传给 ``save_cleaned_data``，
    是为了让测试通过 monkeypatch 修改 ``CLEAN_PATH`` 后能生效
    （Python 函数默认值在定义时绑定，不重新读取模块属性）。

    返回
    -------
    pd.DataFrame
        清洗后的数据，等同于 ``clean_data(load_raw_data())``。
    """
    raw = load_raw_data()
    cleaned = clean_data(raw)
    save_cleaned_data(cleaned, CLEAN_PATH)  # 显式传入，使 monkeypatch 生效
    return cleaned


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_pipeline()