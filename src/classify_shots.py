"""
classify_shots.py — [Membro C]
================================
Classifica ogni colpo rilevato in: dritto, rovescio o servizio.

Due approcci:
  Passo A — Regole if/else (trasparente, spiegabile all'esame)
  Passo B — Random Forest (scikit-learn, confronto con le regole)

Feature calcolate nella finestra ±0.5s attorno all'impatto:
  - Posizione del polso dx rispetto al centro delle anche
  - Direzione orizzontale del polso durante lo swing
  - Altezza del polso rispetto al naso
  - Distanza tra i due polsi (indicatore rovescio a due mani)
  - Braccio sinistro alzato prima dell'impatto (lancio di palla → servizio)

Uso:
    python src/classify_shots.py --keypoints data/keypoints/clip1_clean.csv
                                  --shots data/shots/clip1_shots.csv
                                  --annotations data/annotations/clip1.csv
                                  --output data/shots/clip1_classified.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_features_for_shot(df: pd.DataFrame, impact_frame: int,
                               window_frames: int) -> dict:
    """
    Estrae le feature cinematiche nella finestra ±window_frames attorno all'impatto.
    Tutte le coordinate sono già normalizzate sul busto (da preprocess.py).
    """
    start = max(0, impact_frame - window_frames)
    end = min(len(df) - 1, impact_frame + window_frames)
    window = df.iloc[start:end + 1]

    # Valori all'impatto (frame più vicino)
    impact_idx = df.index[df["frame"] == impact_frame]
    if len(impact_idx) == 0:
        return None
    row = df.loc[impact_idx[0]]

    features = {}

    # 1. Posizione polso dx rispetto al centro delle anche
    features["wrist_r_x"] = row["kp10_x"]   # positivo = destra, negativo = sinistra
    features["wrist_r_y"] = row["kp10_y"]   # negativo = sopra la testa

    # 2. Direzione orizzontale dello swing (pendenza x del polso)
    wrist_x = window["kp10_x"].dropna()
    if len(wrist_x) > 2:
        dx = wrist_x.iloc[-1] - wrist_x.iloc[0]
        features["swing_direction"] = dx  # positivo = da sx a dx (dritto destrorso)
    else:
        features["swing_direction"] = np.nan

    # 3. Altezza polso rispetto al naso (kp0)
    features["wrist_above_nose"] = row["kp0_y"] - row["kp10_y"]   # positivo = polso sopra naso

    # 4. Distanza tra i due polsi (indicatore rovescio a due mani)
    dist_x = row["kp10_x"] - row["kp9_x"]
    dist_y = row["kp10_y"] - row["kp9_y"]
    features["bimanual_dist"] = np.sqrt(dist_x ** 2 + dist_y ** 2)

    # 5. Polso sinistro alzato prima dell'impatto (lancio di palla → servizio)
    left_wrist_before = window[window["frame"] <= impact_frame]["kp9_y"]
    features["left_wrist_raised"] = float(
        left_wrist_before.min() < -0.5 if len(left_wrist_before) > 0 else False
    )

    # 6. Larghezza spalle (si stringe quando il busto ruota)
    features["shoulder_width"] = abs(row["kp5_x"] - row["kp6_x"])

    return features


def classify_rule_based(features: dict, config: dict) -> str:
    """
    Passo A: Classificazione con regole if/else trasparenti.
    Basate sulle misure del piano di lavoro.
    """
    cfg = config["classification"]["rules"]

    # Regola 1: Polso molto sopra la testa → Servizio o Smash
    if features.get("wrist_above_nose", 0) > abs(cfg["wrist_above_head_threshold"]):
        # Braccio sinistro alzato prima → lancio di palla → Servizio
        if features.get("left_wrist_raised", 0) > 0.5:
            return "servizio"
        else:
            return "smash"

    # Regola 2: Entrambi i polsi vicini → Rovescio a due mani
    if features.get("bimanual_dist", 1.0) < cfg["bimanual_dist_threshold"]:
        return "rovescio"

    # Regola 3: Swing da sinistra a destra → Dritto (per Sinner destrorso)
    if features.get("swing_direction", 0) > 0:
        return "dritto"
    else:
        return "rovescio"


def build_feature_matrix(df_kp: pd.DataFrame, shots_df: pd.DataFrame,
                          config: dict) -> tuple[pd.DataFrame, list]:
    """Costruisce la matrice di feature per tutti i colpi."""
    fps = config["statistics"]["fps"]
    window_sec = config["classification"]["window_sec"]
    window_frames = int(window_sec * fps)

    rows = []
    valid_indices = []

    for idx, shot_row in shots_df.iterrows():
        frame = int(shot_row["frame"])
        feats = extract_features_for_shot(df_kp, frame, window_frames)
        if feats is not None:
            rows.append(feats)
            valid_indices.append(idx)

    return pd.DataFrame(rows), valid_indices


def classify_shots(keypoints_path: str, shots_path: str,
                   annotations_path: str, output_path: str, config: dict):

    print(f"📂 Carico keypoints: {keypoints_path}")
    df_kp = pd.read_csv(keypoints_path)

    print(f"📂 Carico colpi rilevati: {shots_path}")
    shots_df = pd.read_csv(shots_path)

    print("🔧 Estraggo feature per ogni colpo...")
    feature_matrix, valid_indices = build_feature_matrix(df_kp, shots_df, config)
    shots_df = shots_df.loc[valid_indices].reset_index(drop=True)

    # ── Passo A: Regole if/else ──────────────────────────────────────
    print("\n📏 Passo A — Classificazione con regole if/else...")
    shots_df["tipo_regole"] = [
        classify_rule_based(row.to_dict(), config)
        for _, row in feature_matrix.iterrows()
    ]
    print(shots_df["tipo_regole"].value_counts().to_string())

    # ── Passo B: Random Forest (solo se le annotazioni esistono) ──────
    if annotations_path and os.path.exists(annotations_path):
        print("\n🌲 Passo B — Random Forest con scikit-learn...")
        annotations = pd.read_csv(annotations_path)  # colonne: frame, tipo

        # Merge colpi con le annotazioni più vicine
        tolerance = config["detection"]["tolerance_frames"]
        labels = []
        for _, shot_row in shots_df.iterrows():
            frame = int(shot_row["frame"])
            close = annotations[abs(annotations["frame"] - frame) <= tolerance]
            if len(close) > 0:
                labels.append(close.iloc[0]["tipo"])
            else:
                labels.append(None)

        shots_df["tipo_annotato"] = labels
        mask = shots_df["tipo_annotato"].notna()
        print(f"   Colpi con etichetta: {mask.sum()} / {len(shots_df)}")

        if mask.sum() >= 10:
            X = feature_matrix[mask].fillna(0)
            y = shots_df.loc[mask, "tipo_annotato"]

            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(clf, X, y_enc, cv=3, scoring="accuracy")
            print(f"   Accuracy CV (3-fold): {scores.mean():.3f} ± {scores.std():.3f}")

            clf.fit(X, y_enc)
            shots_df["tipo_rf"] = le.inverse_transform(clf.predict(feature_matrix.fillna(0)))
        else:
            print("   ⚠️  Troppi pochi colpi annotati per il training — salto il RF.")
    else:
        print("ℹ️  Nessun file di annotazione — salto il Passo B (RF).")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shots_df.to_csv(output_path, index=False)
    print(f"\n✅ Classificazione salvata in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Classifica i colpi in dritto/rovescio/servizio")
    parser.add_argument("--keypoints", required=True, help="CSV keypoints pulito")
    parser.add_argument("--shots", required=True, help="CSV colpi rilevati")
    parser.add_argument("--annotations", default=None, help="CSV annotazioni manuali (per RF)")
    parser.add_argument("--output", required=True, help="CSV colpi classificati")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    classify_shots(args.keypoints, args.shots, args.annotations, args.output, config)


if __name__ == "__main__":
    main()
