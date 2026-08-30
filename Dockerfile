# ─────────────────────────────────────────────────────────────────
# IoT-Sensor-Forecast · Flask API 容器镜像
# 基座：python:3.11-slim（CPU 推理够用，镜像 ~1.5GB）
# 用途：演示场景一键启动 API，CPU 推理不依赖 GPU
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Python 环境变量（生产级最佳实践）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IOT_PROJECT_ROOT=/app \
    PORT=5000

WORKDIR /app

# ─────────────── 系统依赖 ───────────────
# curl 用于 HEALTHCHECK；slim 已自带大部分 Python 依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─────────────── Python 依赖（先 COPY requirements 利用缓存） ───────────────
# flask / flask-cors / gunicorn / pydantic 均在 requirements.txt，
# 这里只装 requirements，避免重复。
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ─────────────── 应用代码（按变更频率从低到高 COPY） ───────────────
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/

# data / models / reports 用 volumes 挂载（避免镜像臃肿）
# 但仍 COPY 空目录结构，让容器启动时不会因目录缺失报错
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/

# 确保目录存在（即使为空）
RUN mkdir -p /app/logs

# ─────────────── 非 root 用户（Docker 安全最佳实践） ───────────────
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# ─────────────── 健康检查（容器编排系统探测用） ───────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:5000/health || exit 1

# ─────────────── 启动 gunicorn 生产服务器 ───────────────
# workers=2：CPU 核数 2-4 时标准配置
# timeout=120：LSTM/Transformer 冷启动推理预留时间
# access-logfile=-：日志输出到 stdout，便于 docker logs 查看
CMD ["gunicorn", "api.flask_app:app", \
     "-b", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]