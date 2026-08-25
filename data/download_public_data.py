from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)


def get_indoor_air_quality():
    ds = fetch_ucirepo(id=438)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(RAW / "uci_indoor_air_quality.csv", index=False)
    print(f"[UCI IAQ] {df.shape}, columns: {list(df.columns)}")
    return df


def get_sml2010():
    ds = fetch_ucirepo(id=275)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(RAW / "sml2010.csv", index=False)
    print(f"[SML2010] {df.shape}")
    return df


if __name__ == "__main__":
    get_indoor_air_quality()
    get_sml2010()
