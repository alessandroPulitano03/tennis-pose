"""
preprocess.py — [Membro A]
===========================
Pulisce e normalizza il CSV grezzo dei keypoint:
  1. Imposta a NaN i keypoint con confidenza < soglia
  2. Interpola i buchi brevi (≤ N frame)
  3. Applica filtro Savitzky-Golay per smussare il rumore
  4. Normalizza le coordinate rispetto all'anca (per rendere la
     pipeline indipendente dalla posizione del giocatore nel frame)

Uso:
    python src/preprocess.py --input data/keypoints/clip1.csv --output data/keypoints/clip1_clean.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_kp_columns(df: pd.DataFrame) -> dict:
    """Restituisce dizionari {kp_id: (col_x, col_y, col_conf)}."""
    cols = {}
    for i in range(17):
        cols[i] = (f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf")
    return cols


def apply_confidence_mask(df: pd.DataFrame, threshold: float, kp_cols: dict) -> pd.DataFrame:
    """Imposta x, y = NaN dove la confidenza è sotto la soglia."""
    df = df.copy()
    for i, (col_x, col_y, col_conf) in kp_cols.items():
        low_conf = df[col_conf] < threshold
        df.loc[low_conf, col_x] = np.nan
        df.loc[low_conf, col_y] = np.nan
    return df


def interpolate_gaps(df: pd.DataFrame, max_gap: int, kp_cols: dict) -> pd.DataFrame:
    """Interpolazione lineare per buchi ≤ max_gap frame."""
    df = df.copy()
    coord_cols = []
    for i, (col_x, col_y, col_conf) in kp_cols.items():
        coord_cols += [col_x, col_y]

    for col in coord_cols:
        df[col] = df[col].interpolate(
            method="linear",
            limit=max_gap,
            limit_direction="both"
        )
    return df


def apply_savgol(df: pd.DataFrame, window: int, poly: int, kp_cols: dict) -> pd.DataFrame:
    """Applica il filtro Savitzky-Golay su x e y di ogni keypoint."""
    df = df.copy()
    for i, (col_x, col_y, col_conf) in kp_cols.items():
        for col in [col_x, col_y]:
            valid = df[col].notna()
            if valid.sum() > window:
                df.loc[valid, col] = savgol_filter(
                    df.loc[valid, col].values,
                    window_length=window,
                    polyorder=poly
                )
    return df


def normalize_by_torso(df: pd.DataFrame, kp_cols: dict) -> pd.DataFrame:
    """
    Normalizza le coordinate usando come riferimento:
      - Origine: centro delle anche (media tra kp11 e kp12)
      - Scala: lunghezza del busto (distanza tra spalle e anche)

    Questo rende le misure indipendenti dalla posizione e dalla
    grandezza del giocatore nell'immagine.
    """
    df = df.copy()

    # Centro anca (origine)
    hip_cx = (df["kp11_x"] + df["kp12_x"]) / 2
    hip_cy = (df["kp11_y"] + df["kp12_y"]) / 2

    # Lunghezza busto (distanza tra centro spalle e centro anche)
    shoulder_cx = (df["kp5_x"] + df["kp6_x"]) / 2
    shoulder_cy = (df["kp5_y"] + df["kp6_y"]) / 2
    torso_len = np.sqrt((shoulder_cx - hip_cx) ** 2 + (shoulder_cy - hip_cy) ** 2)

    # Evita divisioni per zero
    torso_len = torso_len.replace(0, np.nan)

    for i, (col_x, col_y, col_conf) in kp_cols.items():
        df[col_x] = (df[col_x] - hip_cx) / torso_len
        df[col_y] = (df[col_y] - hip_cy) / torso_len

    # Aggiungi colonne di supporto per debug
    df["hip_cx_px"] = hip_cx
    df["hip_cy_px"] = hip_cy
    df["torso_len_px"] = torso_len

    return df


def preprocess(input_path: str, output_path: str, config: dict):
    cfg = config["preprocessing"]

    print(f"📂 Carico: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Righe originali: {len(df)}")

    kp_cols = get_kp_columns(df)

    print("🔧 Step 1: Maschera confidenza bassa...")
    df = apply_confidence_mask(df, cfg["conf_threshold"], kp_cols)

    print("🔧 Step 2: Interpolazione buchi brevi...")
    df = interpolate_gaps(df, cfg["interp_max_gap"], kp_cols)

    print("🔧 Step 3: Filtro Savitzky-Golay...")
    df = apply_savgol(df, cfg["savgol_window"], cfg["savgol_poly"], kp_cols)

    print("🔧 Step 4: Normalizzazione sul busto...")
    df = normalize_by_torso(df, kp_cols)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    nan_pct = df[[f"kp{i}_x" for i in range(17)]].isna().mean().mean() * 100
    print(f"\n✅ CSV pulito salvato in: {output_path}")
    print(f"   Righe: {len(df)} | NaN residui: {nan_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Preprocessa il CSV dei keypoint")
    parser.add_argument("--input", required=True, help="CSV grezzo di input")
    parser.add_argument("--output", required=True, help="CSV pulito di output")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess(args.input, args.output, config)


if __name__ == "__main__":
    main()
