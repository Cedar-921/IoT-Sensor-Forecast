"""IoT-Sensor-Forecast · 模型实现。

对外暴露：
- ARIMAModel：传统时序基线（Day 8）
- XGBoostModel：多变量机器学习（Day 9）
- LSTMModel：双层 LSTM（Day 10-12）
- TransformerModel：Transformer Encoder（Day 13-14）

容错设计：
- LSTMModel / TransformerModel 的导入用 try/except 包裹，
  本地未装 torch 时不抛错（Day 16+ 的 Flask API 才能加载 ARIMA/XGBoost）
- 训练脚本使用前请确认 `import torch` 成功
"""
from src.models.baselines import ARIMAModel
from src.models.xgboost_model import XGBoostModel

try:
    from src.models.lstm_model import LSTMModel
except ImportError as e:
    LSTMModel = None  # type: ignore[assignment,misc]
    _LSTM_IMPORT_ERROR = str(e)

try:
    from src.models.transformer_model import TransformerModel
except ImportError as e:
    TransformerModel = None  # type: ignore[assignment,misc]
    _TRANSFORMER_IMPORT_ERROR = str(e)

__all__ = ["ARIMAModel", "XGBoostModel", "LSTMModel", "TransformerModel"]