"""
detect_shots.py — [Membro B]
==============================
Rileva automaticamente gli istanti di impatto dei colpi analizzando
la velocità del polso destro (keypoint 10) normalizzata sul busto.

Algoritmo:
  1. Calcola la velocità del polso dx (differenza tra frame consecutivi)
  2. Calcola la velocità totale = norma del vettore (vx, vy)
  3. Usa scipy.signal.find_peaks per trovare i picchi di velocità
  4. Confronta con le annotazioni manuali per calcolare precision/recall

Uso:
    python src/detect_shots.py --keypoints data/keypoints/clip1_clean.csv
                               --annotations data/annotations/clip1.csv
                               --output data/shots/clip1_shots.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_wrist_speed(df: pd.DataFrame) -> pd.Series:
    """
    Calcola la velocità del polso destro (kp10) in coordinate normalizzate.
    Velocità = norma del vettore di spostamento frame-per-frame.
    """
    vx = df["kp10_x"].diff()
    vy = df["kp10_y"].diff()
    speed = np.sqrt(vx ** 2 + vy ** 2)
    return speed


def detect_peaks(speed: pd.Series, config: dict, fps: float) -> np.ndarray:
    """
    Trova i picchi di velocità che corrispondono agli impatti.
    Restituisce gli indici di frame dei picchi trovati.
    """
    cfg = config["detection"]

    # Converti distanza minima da secondi a frame
    min_distance_frames = int(cfg["peak_distance_sec"] * fps)

    peaks, properties = find_peaks(
        speed.fillna(0),
        height=cfg["peak_height"],
        distance=min_distance_frames,
    )
    return peaks


def evaluate_detection(detected_frames: np.ndarray,
                        annotations: pd.DataFrame,
                        tolerance: int) -> dict:
    """
    Confronta i frame rilevati con le annotazioni manuali.
    Un rilevamento è corretto se è entro ±tolerance frame da un'annotazione.
    """
    gt_frames = annotations["frame"].values
    matched_gt = set()
    tp = 0

    for det in detected_frames:
        for gt in gt_frames:
            if abs(det - gt) <= tolerance and gt not in matched_gt:
                tp += 1
                matched_gt.add(gt)
                break

    fp = len(detected_frames) - tp
    fn = len(gt_frames) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "n_detected": len(detected_frames),
        "n_annotated": len(gt_frames),
    }


def detect_shots(keypoints_path: str, annotations_path: str,
                 output_path: str, config: dict):
    cfg = config["detection"]
    fps = config["statistics"]["fps"]

    print(f"📂 Carico keypoints: {keypoints_path}")
    df = pd.read_csv(keypoints_path)

    # Calcola velocità polso
    speed = compute_wrist_speed(df)
    df["wrist_speed"] = speed

    # Trova picchi
    peaks = detect_peaks(speed, config, fps)
    print(f"🎯 Picchi di velocità trovati: {len(peaks)}")

    # Costruisci DataFrame dei colpi rilevati
    shots_df = pd.DataFrame({
        "frame": df.loc[peaks, "frame"].values,
        "time_s": df.loc[peaks, "time_s"].values,
        "wrist_speed": speed.iloc[peaks].values,
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shots_df.to_csv(output_path, index=False)
    print(f"✅ Colpi salvati in: {output_path}")

    # Valutazione (solo se le annotazioni esistono)
    if annotations_path and os.path.exists(annotations_path):
        print(f"\n📊 Valutazione vs annotazioni: {annotations_path}")
        annotations = pd.read_csv(annotations_path)
        metrics = evaluate_detection(
            peaks, annotations, cfg["tolerance_frames"]
        )
        print(f"   Rilevati: {metrics['n_detected']} | Annotati: {metrics['n_annotated']}")
        print(f"   TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
        print(f"   Precision={metrics['precision']:.3f} | Recall={metrics['recall']:.3f} | F1={metrics['f1']:.3f}")
    else:
        print("ℹ️  Nessun file di annotazione fornito — salto la valutazione.")


def main():
    parser = argparse.ArgumentParser(description="Rileva automaticamente i colpi dal CSV dei keypoint")
    parser.add_argument("--keypoints", required=True, help="CSV keypoints pulito")
    parser.add_argument("--annotations", default=None, help="CSV annotazioni manuali (opzionale)")
    parser.add_argument("--output", required=True, help="CSV colpi rilevati")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    detect_shots(args.keypoints, args.annotations, args.output, config)


if __name__ == "__main__":
    main()
