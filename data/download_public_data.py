from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# 解析项目根（脚本父目录的父目录），不依赖运行时的 cwd
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = _PROJECT_ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def get_appliances_energy():
    """Appliances Energy Prediction（UCI 374）

    低能耗建筑中 9 个房间的温湿度 + 室外气象，10 分钟采样，
    目标回归预测 Appliances 能耗（Wh）。真实 IoT 传感器场景，
    适配 ARIMA / XGBoost / LSTM / Transformer 4 模型对比。
    """
    ds = fetch_ucirepo(id=374)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(RAW / "appliances_energy.csv", index=False)
    print(f"[Appliances Energy] {df.shape}")
    print(f"  Features: {list(ds.data.features.columns)}")
    print(f"  Targets: {list(ds.data.targets.columns)}")
    return df


if __name__ == "__main__":
    get_appliances_energy()