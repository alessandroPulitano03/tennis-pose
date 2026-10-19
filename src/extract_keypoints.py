"""
extract_keypoints.py — [Membro A]
==================================
Estrae i keypoint del corpo da un video usando YOLO-Pose + tracker ByteTrack.
Richiede un PC con GPU NVIDIA.

Uso:
    python src/extract_keypoints.py --video clips/clip1/video.mp4 --output clips/clip1/keypoints.csv

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


def tracked_boxes(results) -> dict[int, tuple[int, float]]:
    """Persone tracciate nel frame: track_id → (indice nel risultato, area del box)."""
    boxes = {}
    if results.boxes.id is None:
        return boxes
    for i, box in enumerate(results.boxes):
        if box.id is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes[int(box.id.item())] = (i, (x2 - x1) * (y2 - y1))
    return boxes


def choose_player_track(boxes: dict[int, tuple[int, float]]) -> int | None:
    """
    Restituisce il track_id del giocatore vicino alla telecamera: la persona
    con il bounding box più grande (esclude avversario, raccattapalle e giudici).
    """
    if not boxes:
        return None
    return max(boxes, key=lambda tid: boxes[tid][1])


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
        imgsz=cfg_yolo["imgsz"],
    )

    header = build_csv_header()
    player_track_id = None  # Sarà determinato al primo frame
    # Il giocatore viene scelto di nuovo (taglio nel video, cambio di campo,
    # tracker che gli assegna un nuovo ID, persona sbagliata scelta mentre il
    # giocatore era fuori inquadratura) se per più di reselect_after frame:
    #   - non viene più rilevato, oppure
    #   - un'altra persona ha un box più grande di switch_area_ratio volte.
    # reselect_after deve restare > preprocessing.interp_max_gap, così il buco
    # di NaN di un giocatore perso non viene riempito interpolando col successivo.
    reselect_after = round(cfg_yolo["reselect_after_sec"] * fps)
    switch_ratio = cfg_yolo["switch_area_ratio"]
    missing_frames = 0
    bigger_frames = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for frame_idx, result in enumerate(results_gen):
            time_s = round(frame_idx / fps, 4)
            boxes = tracked_boxes(result)

            if player_track_id is not None and missing_frames >= reselect_after:
                print(f"[Frame {frame_idx}] Giocatore track_id={player_track_id} perso"
                      f" da {missing_frames} frame → nuova selezione")
                player_track_id = None

            # Un'altra persona molto più grande di quella seguita → è lei il giocatore
            if player_track_id in boxes:
                largest_id = choose_player_track(boxes)
                if boxes[largest_id][1] > switch_ratio * boxes[player_track_id][1]:
                    bigger_frames += 1
                else:
                    bigger_frames = 0
                if bigger_frames >= reselect_after:
                    print(f"[Frame {frame_idx}] track_id={largest_id} più grande di"
                          f" track_id={player_track_id} da {bigger_frames} frame → cambio giocatore")
                    player_track_id = largest_id
                    bigger_frames = 0

            # Determina il track_id del giocatore (al primo frame o dopo averlo perso)
            if player_track_id is None and boxes:
                player_track_id = choose_player_track(boxes)
                missing_frames = 0
                bigger_frames = 0
                print(f"[Frame {frame_idx}] Giocatore selezionato: track_id={player_track_id}")

            # Box corrispondente al giocatore
            row_data = None
            if player_track_id in boxes:
                i = boxes[player_track_id][0]
                x1, y1, x2, y2 = result.boxes[i].xyxy[0].tolist()

                # Estrai i 17 keypoint (x, y, conf)
                kps = result.keypoints.data[i].cpu().numpy()  # (17, 3)
                kp_flat = kps.flatten().tolist()

                row_data = [frame_idx, time_s, player_track_id,
                            round(x1, 1), round(y1, 1),
                            round(x2, 1), round(y2, 1)] + \
                           [round(v, 4) for v in kp_flat]

            # Se il giocatore non è rilevato in questo frame → riga con NaN
            missing_frames = 0 if row_data is not None else missing_frames + 1
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
