#!/usr/bin/env bash
# IoT-Sensor-Forecast · 一键演示 4 模型 /predict
#
# 前置：
#   1. Flask API 已启动：gunicorn api.flask_app:app -b 0.0.0.0:5000
#   2. 4 个模型文件已在 models/ 下：arima_v1.pkl / xgboost_v1.pkl / lstm_v1.pt / transformer_v1.pt
#   3. data/processed/cleaned.csv 存在（/history 端点依赖）
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

# 3. /history 端点拉真实历史（避免把 290 个数塞 shell 变量截断）
echo "[3/5] /history?n=290（拉真实历史）"
HISTORY_FILE=$(mktemp)
curl -s -f "$API_BASE/history?n=290" | python -c "import sys, json; d=json.load(sys.stdin); json.dump(d['appliances'], open('$HISTORY_FILE','w'))"
HISTORY_LEN=$(python -c "import json; print(len(json.load(open('$HISTORY_FILE'))))")
echo "  history 长度: $HISTORY_LEN"
echo

# 4. 4 模型 /predict（用临时文件写 body，绕过 shell 参数长度限制）
for MODEL in arima xgboost lstm transformer; do
    echo "[4/5] /predict  model=$MODEL"
    if [ "$MODEL" = "arima" ]; then
        BODY=$(printf '{"model":"%s","n_steps":%d}' "$MODEL" "$N_STEPS")
    else
        # 从 history 文件读出数组，嵌入 JSON body
        BODY=$(python -c "
import json
history = json.load(open('$HISTORY_FILE'))
body = {'model': '$MODEL', 'n_steps': $N_STEPS, 'history': history}
print(json.dumps(body))
")
    fi
    BODY_FILE=$(mktemp)
    printf '%s' "$BODY" > "$BODY_FILE"

    RESP_FILE=$(mktemp)
    curl -s -X POST "$API_BASE/predict" \
        -H "Content-Type: application/json" \
        --data-binary "@$BODY_FILE" > "$RESP_FILE"
    rm -f "$BODY_FILE"

    python -c "
import json
d = json.load(open('$RESP_FILE'))
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
    rm -f "$RESP_FILE"
    echo
done

rm -f "$HISTORY_FILE"

echo "[5/5] 完成"
echo "============================================================"
echo "下一步：浏览器打开 dashboard/"
echo "  python -m http.server 8080 --directory dashboard"
echo "  -> http://localhost:8080"
echo "============================================================"