#!/usr/bin/env bash
# IoT-Sensor-Forecast · 一键演示 4 模型 /predict
#
# 前置：
#   1. Flask API 已启动：python api/flask_app.py 或 gunicorn api.flask_app:app
#   2. 4 个模型文件已在 models/ 下：arima_v1.pkl / xgboost_v1.pkl / lstm_v1.pt / transformer_v1.pt
#
# 用法：
#   bash scripts/demo_predict.sh                    # 默认 http://localhost:5000
#   API_BASE=http://192.168.1.5:5000 bash scripts/demo_predict.sh

set -e

# Windows Git Bash 输出 UTF-8（避免 emoji 触发 GBK 错误）
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8

API_BASE="${API_BASE:-http://localhost:5000}"
N_STEPS="${N_STEPS:-12}"  # 预测步数（10min × 12 = 2h）

echo "============================================================"
echo "IoT-Sensor-Forecast · /predict 演示"
echo "API:     $API_BASE"
echo "n_steps: $N_STEPS"
echo "============================================================"
echo

# 1. 健康检查
echo "[1/5] /health"
curl -s -f "$API_BASE/health" | python -m json.tool
echo

# 2. metrics
echo "[2/5] /metrics"
curl -s -f "$API_BASE/metrics" | python -m json.tool
echo

# 3. 生成 290 个 Appliances 历史点（lag288 暖机需要）
#    真实场景应从 cleaned.csv 取最近 290 行；演示用 sin 波模拟
echo "[3/5] 构造 history (290 个 Appliances 假数据)"
HISTORY=$(PYTHONIOENCODING=utf-8 python -c "
import math, json
history = [60 + 30 * math.sin(i / 12) + 20 * math.sin(i / 144) for i in range(290)]
print(json.dumps(history))
")
echo "  history 长度: $(echo "$HISTORY" | python -c 'import sys, json; print(len(json.load(sys.stdin)))')"
echo

# 4. 4 模型 /predict
for MODEL in arima xgboost lstm transformer; do
    echo "[4/5] /predict  model=$MODEL"
    if [ "$MODEL" = "arima" ]; then
        # ARIMA 不需要 history
        RESP=$(curl -s -X POST "$API_BASE/predict" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"n_steps\":$N_STEPS}")
    else
        # XGBoost/LSTM/Transformer 需要 290+ 历史点（演示简化，ARIMA 用模型自带历史）
        RESP=$(curl -s -X POST "$API_BASE/predict" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL\",\"n_steps\":$N_STEPS,\"history\":$HISTORY}")
    fi
    echo "$RESP" | PYTHONIOENCODING=utf-8 python -c "
import sys, json
d = json.load(sys.stdin)
model_name = d.get('model', '$MODEL')
if d.get('status') == 'ok':
    preds = d['predictions']
    print(f'  [OK] {model_name}: {len(preds)} predictions')
    print(f'    first 3: {[round(p,1) for p in preds[:3]]}')
    print(f'    last  3: {[round(p,1) for p in preds[-3:]]}')
    print(f'    elapsed: {d[\"metadata\"][\"elapsed_ms\"]}ms')
else:
    detail = d.get('detail','')[:120]
    err = d.get('error','unknown')
    print(f'  [ERR] {model_name}: {err} -- {detail}')
"
    echo
done

echo "[5/5] 完成"
echo "============================================================"
echo "下一步：浏览器打开 dashboard/"
echo "  python -m http.server 8080 --directory dashboard"
echo "  -> http://localhost:8080"
echo "============================================================"