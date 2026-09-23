"""
render_video.py — [Membro A / Membro C]
=========================================
Genera il video annotato con:
  - Scheletro YOLO-Pose disegnato su ogni frame
  - Etichetta del tipo di colpo visualizzata attorno all'impatto
  - Bounding box del giocatore

Uso:
    python src/render_video.py --video data/raw/clip1.mp4
                                --keypoints data/keypoints/clip1_clean.csv
                                --classified data/shots/clip1_classified.csv
                                --output results/video/clip1_annotated.mp4
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
import yaml

# Connessioni dello scheletro COCO (coppie di keypoint da collegare)
SKELETON = [
    (5, 6),   # spalle
    (5, 7), (7, 9),   # braccio sinistro
    (6, 8), (8, 10),  # braccio destro
    (5, 11), (6, 12), # busto
    (11, 12),          # fianchi
    (11, 13), (13, 15), # gamba sinistra
    (12, 14), (14, 16), # gamba destra
]

COLOR_MAP = {
    "dritto":   (255, 100, 50),   # arancione (BGR)
    "rovescio": (50, 100, 255),   # rosso/viola
    "servizio": (50, 200, 50),    # verde
    "smash":    (50, 200, 255),   # giallo
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def draw_skeleton(frame: np.ndarray, kps: np.ndarray,
                  color: tuple, thickness: int) -> np.ndarray:
    """
    Disegna lo scheletro sul frame.
    kps: array (17, 3) — x, y, conf (coordinate in pixel, già denormalizzate)
    """
    for i, (x, y, c) in enumerate(kps):
        if c > 0 and not np.isnan(x):
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    for (a, b) in SKELETON:
        xa, ya, ca = kps[a]
        xb, yb, cb = kps[b]
        if ca > 0 and cb > 0 and not np.isnan(xa) and not np.isnan(xb):
            cv2.line(frame,
                     (int(xa), int(ya)),
                     (int(xb), int(yb)),
                     color, thickness)
    return frame


def render_video(video_path: str, keypoints_path: str, classified_path: str,
                 output_path: str, config: dict):
    cfg_render = config["render"]
    fps = config["statistics"]["fps"]

    print(f"📂 Carico video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📂 Carico keypoints: {keypoints_path}")
    df_kp = pd.read_csv(keypoints_path)

    print(f"📂 Carico colpi classificati: {classified_path}")
    shots_df = pd.read_csv(classified_path)
    tipo_col = "tipo_rf" if "tipo_rf" in shots_df.columns else "tipo_regole"

    # Costruisci dizionario frame → tipo colpo
    label_duration_frames = int(video_fps * 1.5)  # mostra l'etichetta per 1.5 secondi
    frame_labels = {}
    for _, row in shots_df.iterrows():
        impact_frame = int(row["frame"])
        tipo = row.get(tipo_col, "dritto")
        for f in range(impact_frame, impact_frame + label_duration_frames):
            frame_labels[f] = tipo

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    # Indice keypoint per frame
    df_kp_indexed = df_kp.set_index("frame")

    frame_idx = 0
    print("🎬 Rendering in corso...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recupera i keypoint di questo frame
        if frame_idx in df_kp_indexed.index:
            row = df_kp_indexed.loc[frame_idx]

            # Denormalizza: riconverti le coordinate normalizzate in pixel
            hip_cx = row.get("hip_cx_px", np.nan)
            hip_cy = row.get("hip_cy_px", np.nan)
            torso_len = row.get("torso_len_px", np.nan)

            kps = np.zeros((17, 3))
            for i in range(17):
                x_norm = row.get(f"kp{i}_x", np.nan)
                y_norm = row.get(f"kp{i}_y", np.nan)
                conf   = row.get(f"kp{i}_conf", 0)
                if not np.isnan(x_norm) and not np.isnan(torso_len):
                    kps[i] = [x_norm * torso_len + hip_cx,
                               y_norm * torso_len + hip_cy,
                               conf]
                else:
                    kps[i] = [np.nan, np.nan, 0]

            skel_color = tuple(cfg_render["skeleton_color"])
            frame = draw_skeleton(frame, kps, skel_color, cfg_render["thickness"])

            # Bounding box giocatore
            x1 = row.get("bbox_x1", np.nan)
            y1 = row.get("bbox_y1", np.nan)
            x2 = row.get("bbox_x2", np.nan)
            y2 = row.get("bbox_y2", np.nan)
            if not np.isnan(x1):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (200, 200, 200), 1)

        # Etichetta colpo
        if frame_idx in frame_labels:
            tipo = frame_labels[frame_idx]
            label_color = COLOR_MAP.get(tipo, (255, 255, 255))
            cv2.putText(frame, tipo.upper(),
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        cfg_render["font_scale"],
                        label_color,
                        cfg_render["thickness"] + 1,
                        cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 200 == 0:
            print(f"  → Frame {frame_idx}")

    cap.release()
    out.release()
    print(f"\n✅ Video annotato salvato in: {output_path}")
    print(f"   Totale frame: {frame_idx}")


def main():
    parser = argparse.ArgumentParser(description="Genera il video annotato con scheletro ed etichette")
    parser.add_argument("--video", required=True, help="Video originale (.mp4)")
    parser.add_argument("--keypoints", required=True, help="CSV keypoints pulito")
    parser.add_argument("--classified", required=True, help="CSV colpi classificati")
    parser.add_argument("--output", required=True, help="Video di output annotato")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    render_video(args.video, args.keypoints, args.classified, args.output, config)


if __name__ == "__main__":
    main()
