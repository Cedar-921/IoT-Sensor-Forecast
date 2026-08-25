# IoT-Sensor-Forecast

IoT 传感器时序数据预测，对比 ARIMA / XGBoost / LSTM / Transformer 四个模型。

## 项目结构

```
IoT-Sensor-Forecast/
├── data/
│   ├── raw/               # 原始数据集（不提交到 git）
│   ├── processed/         # 清洗后数据（不提交到 git）
│   └── download_public_data.py  # 下载 UCI 公开数据集
├── src/
│   ├── feature_engineering.py   # 特征工程
│   └── models/                  # 模型实现
├── api/
│   └── flask_app.py             # Flask REST API
├── dashboard/
│   └── templates/index.html     # ECharts 仪表盘
├── notebooks/
│   └── 01_eda.ipynb             # 数据探索
├── tests/
│   └── test_features.py         # 单元测试
├── configs/
│   └── config.yaml              # 配置文件
├── requirements.txt
└── Dockerfile
```

## 快速开始

```bash
# 克隆
git clone git@github.com:Cedar-921/IoT-Sensor-Forecast.git
cd IoT-Sensor-Forecast

# 建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows

# 装依赖
pip install -r requirements.txt

# 下载数据
python data/download_public_data.py

# 训练模型
python src/train.py --model lstm

# 启动 API
python api/flask_app.py
```

## 数据集

| 数据集 | 来源 | 描述 |
|---|---|---|
| Indoor Air Quality | UCI ML Repo (id=438) | 室内温湿度+CO2，5min采样，~5万条 |
| SML2010 | UCI ML Repo (id=275) | 清华机房温度预测，1min采样 |

## 模型对比

| 模型 | 特点 |
|---|---|
| ARIMA | 传统时序基线 |
| XGBoost | 表格特征+滑窗 |
| LSTM | 时序依赖建模 |
| Transformer | 自注意力机制 |
