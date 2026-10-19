"""
classify_shots.py — [Membro C]
================================
Classifica ogni colpo rilevato in: dritto, rovescio, servizio, smash o volee
con regole if/else (trasparenti, spiegabili all'esame).

Feature calcolate nella finestra ±0.5s attorno all'impatto:
  - Posizione del polso dx rispetto al centro delle anche
  - Lato della preparazione (polso dx a destra o a sinistra delle anche)
  - Direzione orizzontale del polso durante lo swing
  - Altezza del polso rispetto alle spalle (il naso non è visibile da dietro)
  - Distanza tra i due polsi (indicatore rovescio a due mani)
  - Braccio sinistro alzato prima dell'impatto (lancio di palla → servizio)
  - Ampiezza dello swing (la volee è un colpo corto, quasi senza apertura)
  - Vicinanza alla rete: altezza dei piedi nell'immagine rispetto al resto
    della clip (i piedi poggiano sul campo: a rete salgono nell'immagine)

Uso:
    python src/classify_shots.py --keypoints clips/clip1/keypoints_clean.csv
                                  --shots clips/clip1/shots.csv
                                  --output clips/clip1/classified.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_features_for_shot(df: pd.DataFrame, impact_frame: int,
                               window_frames: int, ref: dict) -> dict:
    """
    Estrae le feature cinematiche nella finestra ±window_frames attorno all'impatto.
    Tutte le coordinate sono già normalizzate sul busto (da preprocess.py).
    ref contiene i valori tipici della clip in pixel (busto, altezza caviglie).
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
    before = window[window["frame"] <= impact_frame]

    # Altezza del centro spalle: da dietro il naso non si vede, quindi le
    # spalle fanno da riferimento per "sopra la testa" (y cresce verso il basso)
    shoulder_y = (window["kp5_y"] + window["kp6_y"]) / 2

    # 1. Posizione polso dx rispetto al centro delle anche
    features["wrist_r_x"] = row["kp10_x"]   # positivo = destra, negativo = sinistra
    features["wrist_r_y"] = row["kp10_y"]   # negativo = sopra le anche

    # 2. Lato della preparazione: posizione media del polso dx prima dell'impatto.
    #    Destrorso visto da dietro: dritto preparato a destra (> 0),
    #    rovescio preparato a sinistra (< 0)
    features["prep_side"] = before["kp10_x"].mean()

    # 3. Direzione orizzontale dello swing (pendenza x del polso)
    wrist_x = window["kp10_x"].dropna()
    if len(wrist_x) > 2:
        dx = wrist_x.iloc[-1] - wrist_x.iloc[0]
        features["swing_direction"] = dx  # negativo = da dx a sx (dritto destrorso)
    else:
        features["swing_direction"] = np.nan

    # 4. Massima altezza del polso dx sopra le spalle nella finestra
    features["wrist_above_shoulder"] = (shoulder_y - window["kp10_y"]).max()

    # 5. Distanza tra i due polsi (indicatore rovescio a due mani)
    dist_x = row["kp10_x"] - row["kp9_x"]
    dist_y = row["kp10_y"] - row["kp9_y"]
    features["bimanual_dist"] = np.sqrt(dist_x ** 2 + dist_y ** 2)

    # 6. Massima altezza del polso sx sopra le spalle prima dell'impatto
    #    (lancio di palla → servizio)
    features["left_wrist_above_shoulder"] = (
        shoulder_y.loc[before.index] - before["kp9_y"]
    ).max()

    # 7. Vicinanza alla rete. La telecamera è fissa dietro il giocatore e i
    #    piedi poggiano sul piano del campo: più il giocatore avanza verso la
    #    rete, più i piedi salgono nell'immagine (indipendentemente dalla
    #    postura). Si misura quanto le caviglie sono più in alto della loro
    #    posizione tipica nella clip, in lunghezze di busto tipiche.
    #    Si usa il punto più basso nella finestra, così il salto del servizio
    #    o dello smash non conta. ~0 = fondo campo, valori grandi = a rete.
    ankle_y_px = ((window["kp15_y"] + window["kp16_y"]) / 2
                  * window["torso_len_px"] + window["hip_cy_px"])
    features["feet_advance"] = (ref["ankle_y_px"] - ankle_y_px.max()) / ref["torso_len_px"]

    # 8. Larghezza spalle (si stringe quando il busto ruota)
    features["shoulder_width"] = abs(row["kp5_x"] - row["kp6_x"])

    # 9. Ampiezza dello swing: escursione orizzontale del polso dx prima
    #    dell'impatto. La volee è un colpo "bloccato", quasi senza apertura;
    #    dritto e rovescio (anche giocati vicino alla rete) hanno uno swing ampio
    features["swing_range"] = before["kp10_x"].max() - before["kp10_x"].min()

    return features


def classify_rule_based(features: dict, config: dict) -> str:
    """
    Passo A: Classificazione con regole if/else trasparenti.
    Basate sulle misure del piano di lavoro.
    """
    cfg = config["classification"]["rules"]
    wrist_high = features.get("wrist_above_shoulder", 0) > cfg["wrist_above_shoulder_threshold"]

    # Regola 1: Giocatore vicino alla rete
    if features.get("feet_advance", 0) > cfg["net_feet_advance_threshold"]:
        # Polso sopra le spalle → Smash
        if wrist_high:
            return "smash"
        # Swing corto → Volee. Con uno swing ampio è un colpo da fondo giocato
        # vicino alla rete (es. recupero di una palla corta): regole successive
        if features.get("swing_range", 0) < cfg["volley_swing_range_threshold"]:
            return "volee"

    # Regola 2: Da fondo campo, polso sopra le spalle e braccio sinistro
    # alzato prima (lancio di palla) → Servizio. Senza lancio è la chiusura
    # alta di un dritto, che prosegue nelle regole successive.
    if wrist_high and features.get("left_wrist_above_shoulder", 0) > cfg["left_wrist_raise_threshold"]:
        return "servizio"

    # Regola 3: Polsi molto distanti → Dritto. Nel dritto il braccio sinistro
    # si apre per equilibrio; nel rovescio la mano sx impugna la racchetta (a
    # due mani) o resta vicina al corpo (a una mano). Utile quando il lato
    # della preparazione è ambiguo (es. dritto colpito arretrando)
    if features.get("bimanual_dist", 0) > cfg["forehand_bimanual_dist_threshold"]:
        return "dritto"

    # Regola 4: Entrambi i polsi vicini → Rovescio a due mani
    if features.get("bimanual_dist", 1.0) < cfg["bimanual_dist_threshold"]:
        return "rovescio"

    # Regola 5: Preparazione a destra delle anche → Dritto (Sinner è destrorso)
    if features.get("prep_side", 0) > 0:
        return "dritto"
    else:
        return "rovescio"


def build_feature_matrix(df_kp: pd.DataFrame, shots_df: pd.DataFrame,
                          config: dict) -> tuple[pd.DataFrame, list]:
    """Costruisce la matrice di feature per tutti i colpi."""
    fps = config["statistics"]["fps"]
    window_sec = config["classification"]["window_sec"]
    window_frames = int(window_sec * fps)
    # Valori tipici in pixel: la mediana sulla clip rappresenta il giocatore a
    # fondo campo, dove passa la maggior parte del tempo
    ankle_y_px = ((df_kp["kp15_y"] + df_kp["kp16_y"]) / 2
                  * df_kp["torso_len_px"] + df_kp["hip_cy_px"])
    ref = {"torso_len_px": df_kp["torso_len_px"].median(),
           "ankle_y_px": ankle_y_px.median()}

    rows = []
    valid_indices = []

    for idx, shot_row in shots_df.iterrows():
        frame = int(shot_row["frame"])
        feats = extract_features_for_shot(df_kp, frame, window_frames, ref)
        if feats is not None:
            rows.append(feats)
            valid_indices.append(idx)

    return pd.DataFrame(rows), valid_indices


def classify_shots(keypoints_path: str, shots_path: str, output_path: str, config: dict):

    print(f"📂 Carico keypoints: {keypoints_path}")
    df_kp = pd.read_csv(keypoints_path)

    print(f"📂 Carico colpi rilevati: {shots_path}")
    shots_df = pd.read_csv(shots_path)

    print("🔧 Estraggo feature per ogni colpo...")
    feature_matrix, valid_indices = build_feature_matrix(df_kp, shots_df, config)
    shots_df = shots_df.loc[valid_indices].reset_index(drop=True)

    print("\n📏 Classificazione con regole if/else...")
    shots_df["tipo_regole"] = [
        classify_rule_based(row.to_dict(), config)
        for _, row in feature_matrix.iterrows()
    ]
    print(shots_df["tipo_regole"].value_counts().to_string())

    # Salva anche le feature: servono per i boxplot e per calibrare le soglie
    shots_df = pd.concat([shots_df, feature_matrix], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shots_df.to_csv(output_path, index=False)
    print(f"\n✅ Classificazione salvata in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Classifica i colpi in dritto/rovescio/servizio/smash/volee")
    parser.add_argument("--keypoints", required=True, help="CSV keypoints pulito")
    parser.add_argument("--shots", required=True, help="CSV colpi rilevati")
    parser.add_argument("--output", required=True, help="CSV colpi classificati")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    classify_shots(args.keypoints, args.shots, args.output, config)


if __name__ == "__main__":
    main()
