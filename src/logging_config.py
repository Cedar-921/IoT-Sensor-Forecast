"""IoT-Sensor-Forecast · 统一日志配置（L-10）。

所有模块（api/、src/）通过 `setup_logging()` 初始化日志，避免
`logging.basicConfig` 在多处重复定义导致格式不一致。

设计要点：
- 单一函数入口，便于测试 monkeypatch
- 时间戳 + 级别 + 模块名 + 消息（与 train.py 原格式一致）
- 级别由 LOG_LEVEL 环境变量控制（默认 INFO）
"""
from __future__ import annotations

import logging
import os

DEFAULT_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    level: str | None = None,
    format_string: str = DEFAULT_FORMAT,
) -> None:
    """初始化全局日志配置。

    参数
    ----------
    level : str | None
        日志级别（DEBUG/INFO/WARNING/ERROR）。None 时读 LOG_LEVEL 环境变量，默认 INFO。
    format_string : str
        日志格式字符串（默认与 train.py 保持一致）。
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level.upper(),
        format=format_string,
        force=True,  # 覆盖已有的 basicConfig（多模块各自 import 时必备）
    )