"""pytest 全局配置与 fixtures。

约定：
- 所有测试在执行前必须确保 data/raw/appliances_energy.csv 与 data/processed/cleaned.csv 存在。
- 若不存在，pytest 在 collection 阶段给出清晰错误，而非大量 import 失败。
- 数据文件由**云端**准备（`python data/download_public_data.py` + `python src/data_cleaning.py`），
  scp 拉回本地 `data/` 后才能跑测试。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data_cleaning import RAW_PATH, CLEAN_PATH


def pytest_configure(config):
    """pytest 启动时检查关键数据文件。"""
    missing = []
    if not Path(RAW_PATH).exists():
        missing.append(f"原始数据：{RAW_PATH}（先在云端跑 python data/download_public_data.py）")
    if not Path(CLEAN_PATH).exists():
        missing.append(f"清洗数据：{CLEAN_PATH}（先在云端跑 python src/data_cleaning.py）")
    if missing:
        # 用 pytest.Exit 友好提示，不抛 traceback
        raise pytest.UsageError(
            "数据文件缺失，请先在云端准备数据再跑测试：\n  "
            + "\n  ".join(missing)
        )