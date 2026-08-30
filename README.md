# IoT-Sensor-Forecast · IoT 传感器时序预测

> 🏠 **低能耗建筑能耗预测** · 4 模型对比 · 端到端 MLOps 工程实践

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org)
[![Docker](https://img.shields.io/badge/Docker-✓-2496ED)](https://www.docker.com)
[![Tests](https://img.shields.io/badge/Tests-66%2B-green)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

一个**端到端**的 IoT 时序预测项目：从公开数据集（UCI 374）出发，经过数据清洗、特征工程、4 个模型对比，最终提供 REST API 与可视化 Dashboard。**重点不是单模型 SOTA，而是工程化全链路**。

---

## 🌟 项目亮点

| 模块 | 实现内容 |
|---|---|
| 📊 **数据** | UCI 374 Appliances Energy（19,735 × 28，10min 采样） |
| 🔧 **特征** | 周期编码 + lag(1,3,6,12,24,288) + rolling(144) 共 **40+ 衍生特征** |
| 🤖 **模型** | ARIMA / XGBoost / LSTM / Transformer **4 模型对比** |
| 🚀 **API** | Flask + gunicorn + flask-cors，支持 `/health` / `/metrics` / `/predict` |
| 📈 **Dashboard** | ECharts 纯静态 HTML + 实时 API 调用 |
| 🐳 **部署** | 单命令 `docker compose up` 启动，CPU 推理 ~1.5GB 镜像 |
| ✅ **测试** | pytest 66+ 用例，覆盖数据清洗、特征工程、模型序列化 |

---

## 📊 模型效果

> 验证集（val 15%，约 2960 条）上的真实指标，来自 `reports/results.csv`（云端训练后归档）。

| 模型 | MAE ↓ | RMSE ↓ | MAPE ↓ | 训练时长 | 推理延迟（CPU） |
|---|---|---|---|---|---|
| ARIMA | 44.19 | 94.25 | 41.41% | ~2 min | <50ms |
| XGBoost | 28.92 | 65.35 | 26.26% | ~5 min | ~10ms |
| **LSTM** | 31.63 | 79.18 | 25.78% | ~15 min | ~2s |
| **Transformer** | **31.41** | **77.32** | 27.52% | ~20 min | ~3s |

**关键观察**：

- XGBoost 在 MAE/RMSE 上最优（梯度提升对结构化特征 + 周期编码天然友好）
- Transformer 略胜 LSTM（注意力 > 循环），MAPE 稍逊（高分位误差敏感）
- ARIMA 基线显著落后（单变量 + 线性，无法捕捉多变量非线性）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    云端 GPU 服务器                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐      │
│  │ Raw CSV  │───▶│ 清洗+特征 │───▶│  4 模型训练      │      │
│  │ (UCI 374)│    │ 工程      │    │ ARIMA/XGB/LSTM/  │      │
│  └──────────┘    └──────────┘    │ Transformer       │      │
│                                   └────────┬─────────┘      │
│                                            ▼                │
│                                   ┌──────────────────┐      │
│                                   │ Flask + gunicorn │      │
│                                   │ /health/metrics/ │      │
│                                   │ /predict         │      │
│                                   └────────┬─────────┘      │
└────────────────────────────────────────────┼────────────────┘
                                             │ HTTP + CORS
                                             ▼
                                ┌──────────────────────┐
                                │  本地浏览器 Dashboard │
                                │  (纯静态 ECharts)    │
                                └──────────────────────┘
```

---

## 📁 项目结构

```
IoT-Sensor-Forecast/
├── src/                          # 核心代码
│   ├── data_cleaning.py          # 5 个清洗函数 + 4 步流水线
│   ├── feature_engineering.py    # 40+ 衍生特征
│   ├── sequence_dataset.py       # 滑窗数据集（LSTM/Transformer 共用）
│   ├── evaluate.py               # MAE / RMSE / MAPE
│   ├── train.py                  # 统一训练入口（4 模型注册）
│   └── models/
│       ├── baselines.py          # ARIMAModel
│       ├── xgboost_model.py      # XGBoostModel
│       ├── lstm_model.py         # LSTMModel
│       └── transformer_model.py  # TransformerModel
├── api/
│   └── flask_app.py              # Flask REST API
├── dashboard/
│   └── templates/index.html      # ECharts Dashboard（纯静态）
├── tests/                        # pytest 66+ 用例
├── configs/
│   └── config.yaml               # 全局配置
├── notebooks/
│   └── 01_eda.ipynb              # 中文 EDA（13 cells，5 张图）
├── data/
│   ├── raw/                      # 原始 CSV（git 忽略）
│   ├── processed/                # 清洗后 CSV（git 忽略）
│   └── download_public_data.py   # UCI 数据下载脚本
├── models/                       # 训练生成（git 忽略）
├── reports/                      # 训练指标 CSV
├── Dockerfile                    # Docker 镜像构建
├── docker-compose.yml            # 单服务编排
├── requirements.txt              # Python 依赖
└── pytest.ini                    # pytest 配置
```

---

## 🚀 快速开始

### 方式 A：Docker 一键启动（**推荐**）

```bash
# 1. 克隆
git clone https://github.com/Cedar-921/IoT-Sensor-Forecast.git
cd IoT-Sensor-Forecast

# 2. 构建并启动
docker compose up -d

# 3. 验证
curl http://localhost:5000/health
# → {"status":"ok","service":"iot-sensor-forecast"}

curl http://localhost:5000/metrics
# → [{"model":"arima","MAE":44.19,"RMSE":94.25,"MAPE":41.41},...]
```

### 方式 B：本地开发（venv）

```bash
# 1. 环境
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
pip install flask-cors

# 2. 启动 API
python api/flask_app.py
# → 监听 http://localhost:5000
```

### 方式 C：云端训练 + 本地 Dashboard（推荐生产部署）

```bash
# === 云端（Ubuntu + NVIDIA GPU）===
ssh user@server
git clone https://github.com/Cedar-921/IoT-Sensor-Forecast.git
cd IoT-Sensor-Forecast

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
nvidia-smi   # 查 CUDA 版本

# CUDA 11.8: pip install torch --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1: pip install torch --index-url https://download.pytorch.org/whl/cu121

# 数据
python data/download_public_data.py
python src/data_cleaning.py

# 后台训练（断线不丢）
mkdir -p logs
nohup python src/train.py --model lstm > logs/lstm_train.log 2>&1 &
nohup python src/train.py --model transformer > logs/transformer_train.log 2>&1 &
nohup gunicorn api.flask_app:app -b 0.0.0.0:5000 --workers 2 > logs/flask.log 2>&1 &

# === 本地（Windows）===
# 改 dashboard/templates/index.html 第 51 行 API_BASE 为云服务器 IP
python -m http.server 8080 --directory dashboard
# 浏览器打开 http://localhost:8080
```

---

## 🔧 技术栈

| 层 | 工具 |
|---|---|
| **数据处理** | pandas / numpy / scikit-learn |
| **时序基线** | statsmodels（ARIMA） |
| **表格模型** | xgboost |
| **深度模型** | PyTorch（LSTM / Transformer） |
| **Web 框架** | Flask + flask-cors + gunicorn |
| **可视化** | ECharts（纯静态 HTML） |
| **部署** | Docker + docker compose |
| **测试** | pytest |

---

## 📝 关键设计决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| **目标变换** | `log1p(Appliances)` | 右偏分布（max 1080 vs median 60），对数空间建模更稳定 |
| **周期编码** | `sin/cos` 替代原始 `hour` | 周期边界平滑（23→0 不跳跃），模型更易学 |
| **lag 防泄漏** | `series.shift(k)` | lag(k) 严格取 t-k 时刻的值，不含当前 |
| **rolling 防泄漏** | `shift(1).rolling(w)` | 先右移 1 步再滚动，确保不包含当前时刻 |
| **LSTM/Transformer 目标** | log1p 空间训练 + expm1 反变换 | 与 XGBoost 对齐，统一目标空间 |
| **LSTM 损失** | `L1Loss`（MAE） | 与评估指标对齐，对异常值鲁棒 |
| **Transformer 位置编码** | 标准正弦（不可学习） | 归纳偏置更强，与原始论文一致 |
| **checkpoint 内容** | state_dict + scaler + feature_names | 防止线上推理列错位事故 |
| **CORS 策略** | 仅允许 `localhost:*` | 演示场景专用，避免公网滥用 |
| **gunicorn workers** | `--workers 2` | CPU 核数 2-4 时标准配置 |

---

## 🧪 测试

```bash
# 全量测试（66+ 用例，CPU 秒过）
pytest tests/ -v

# 单文件
pytest tests/test_cleaning.py -v

# 覆盖率
pytest tests/ --cov=src --cov-report=term-missing
```

测试覆盖：
- ✅ 数据清洗（6 用例：缺失列、date bug、NaT、log 变换、save/load）
- ✅ 特征工程（时间特征、lag、rolling、防泄漏）
- ✅ 序列数据集（窗口对齐、target 对齐）
- ✅ ARIMA / XGBoost（fit / predict / save / load / 形状 / 异常）
- ✅ LSTM / Transformer（fit / predict / save-load 一致性 / GPU 自动选择）

---

## 📦 部署架构

**方案：API 云 + Dashboard 本**

| 组件 | 位置 | 说明 |
|---|---|---|
| 模型训练 | 云端 GPU | RTX 5090 / A100，CUDA 11.8+/12.1+ |
| Flask API | 云端（Docker） | gunicorn 2 workers |
| Dashboard | 本地浏览器 | 纯静态 HTML，CORS 调远程 API |

**为什么这样设计**：
- 模型权重不上传 GitHub（git 忽略 `*.pt/*.pkl`）
- Dashboard 不需要服务器（浏览器 + http.server 即可）
- 一键切换 API 地址（IP / 域名 / ngrok）

---

## 📊 数据集

| 字段 | 详情 |
|---|---|
| **名称** | Appliances Energy Prediction |
| **来源** | [UCI ML Repository id=374](https://archive.ics.uci.edu/dataset/374) |
| **样本** | 19,735 条（10min 采样，约 4.5 个月） |
| **特征** | 28 维：9 房间温湿度 + 室外气象 + 风向/可见度/露点 |
| **目标** | `Appliances` 能耗（Wh），右偏（10~1080，median 60） |

### 关键 EDA 发现（已编码到清洗模块）

1. **T6 是室外探头误标**（与 T_out 相关 0.975）→ 删除
2. **rv1 = rv2 完全冗余**（corr=1.0）→ 删除 rv2
3. **date 字符串缺空格**（`2016-01-1117:00:00`）→ 正则修复
4. **Appliances 严重右偏**（skew=3.386）→ `log1p` 变换
5. **lights 是关键开关**（77.3% 为 0，但相关 top1）→ 保留原值

---

## 🤝 相关链接

- 📓 **ima 笔记**：IoT-Sensor-Forecast 紧凑执行方案（项目执行手册）
- 📊 **数据集**：UCI ML Repository id=374
- 🐍 **Python**：3.11+
- 🐳 **Docker**：24+

---

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

---

> 💡 **面试官快速扫读指引**：
> - 30 秒看 `🌟 项目亮点` + `📊 模型效果`
> - 5 分钟看 `🏗️ 系统架构` + `📝 关键设计决策`
> - 15 分钟跑 `docker compose up` + `pytest tests/ -v`
> - 30 分钟读 `src/train.py` + `src/models/lstm_model.py`（最复杂的两块）