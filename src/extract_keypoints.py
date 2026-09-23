"""
extract_keypoints.py — [Membro A]
==================================
Estrae i keypoint del corpo da un video usando YOLO-Pose + tracker ByteTrack.
Richiede un PC con GPU NVIDIA.

Uso:
    python src/extract_keypoints.py --video data/raw/clip1.mp4 --output data/keypoints/clip1.csv

Output CSV:
    frame, time_s, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    kp0_x, kp0_y, kp0_conf, ..., kp16_x, kp16_y, kp16_conf
"""

import argparse
import csv
import os

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_csv_header() -> list[str]:
    """Costruisce l'intestazione del CSV con le 17 * 3 colonne keypoint."""
    base = ["frame", "time_s", "track_id",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    kp_cols = []
    for i in range(17):
        kp_cols += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"]
    return base + kp_cols


def choose_player_track(results) -> int | None:
    """
    Restituisce il track_id del giocatore Sinner: la persona con il
    bounding box più grande e più in basso nell'immagine (esclude
    avversario, raccattapalle e giudici).
    """
    best_id = None
    best_area = 0

    for box in results.boxes:
        if box.id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_id = int(box.id.item())

    return best_id


def extract_keypoints(video_path: str, output_path: str, config: dict):
    cfg_yolo = config["yolo"]

    model = YOLO(cfg_yolo["model"])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Il tracker mantiene gli ID stabili tra i frame
    results_gen = model.track(
        source=video_path,
        stream=True,
        persist=True,
        device=cfg_yolo["device"],
        conf=cfg_yolo["conf"],
        iou=cfg_yolo["iou"],
    )

    header = build_csv_header()
    player_track_id = None  # Sarà determinato al primo frame

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for frame_idx, result in enumerate(results_gen):
            time_s = round(frame_idx / fps, 4)

            # Determina il track_id del giocatore al primo frame
            if player_track_id is None and result.boxes.id is not None:
                player_track_id = choose_player_track(result)
                print(f"[Frame {frame_idx}] Giocatore selezionato: track_id={player_track_id}")

            # Trova il box corrispondente al giocatore
            row_data = None
            if result.boxes.id is not None:
                for i, box in enumerate(result.boxes):
                    if box.id is None:
                        continue
                    tid = int(box.id.item())
                    if tid == player_track_id:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Estrai i 17 keypoint (x, y, conf)
                        kps = result.keypoints.data[i].cpu().numpy()  # (17, 3)
                        kp_flat = kps.flatten().tolist()

                        row_data = [frame_idx, time_s, tid,
                                    round(x1, 1), round(y1, 1),
                                    round(x2, 1), round(y2, 1)] + \
                                   [round(v, 4) for v in kp_flat]
                        break

            # Se il giocatore non è rilevato in questo frame → riga con NaN
            if row_data is None:
                nan_row = [frame_idx, time_s,
                           player_track_id if player_track_id else -1,
                           np.nan, np.nan, np.nan, np.nan]
                nan_row += [np.nan] * (17 * 3)
                row_data = nan_row

            writer.writerow(row_data)

            if frame_idx % 100 == 0:
                print(f"  → Frame {frame_idx} | time={time_s:.2f}s")

    print(f"\n✅ CSV salvato in: {output_path}")
    print(f"   Totale frame: {frame_idx + 1}")


def main():
    parser = argparse.ArgumentParser(description="Estrai keypoint YOLO-Pose da un video di tennis")
    parser.add_argument("--video", required=True, help="Percorso al video di input (.mp4)")
    parser.add_argument("--output", required=True, help="Percorso al CSV di output")
    parser.add_argument("--config", default="config.yaml", help="File di configurazione")
    args = parser.parse_args()

    config = load_config(args.config)
    extract_keypoints(args.video, args.output, config)


if __name__ == "__main__":
    main()
