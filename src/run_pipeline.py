"""
run_pipeline.py
================
Esegue l'intera pipeline su un video, tenendo tutti i file di quel video
in una cartella dedicata: clips/<nome_clip>/

    clips/<nome_clip>/
    ├── video.<ext>            # copia del video originale
    ├── annotations.csv        # scritto a mano (frame, tipo)
    ├── config.yaml            # parametri specifici della clip (opzionale)
    ├── config_usato.yaml      # config effettivo usato (generato)
    ├── keypoints.csv          # extract_keypoints.py
    ├── keypoints_clean.csv    # preprocess.py
    ├── shots.csv              # detect_shots.py
    ├── classified.csv         # classify_shots.py
    └── results/
        ├── evaluation/        # evaluate.py (solo con annotations.csv)
        ├── statistics/        # game_stats.py
        ├── plots/             # visualize.py
        └── annotated.mp4      # render_video.py

Uno step viene rieseguito solo se il suo output manca o è più vecchio
di uno dei suoi input (es. dopo aver aggiunto annotations.csv vengono
rifatti rilevamento, valutazione e output finali, ma non YOLO).

Uso:
    # Nuovo video: crea clips/<nome_file>/ e lancia tutto
    python src/run_pipeline.py --video C:/percorso/alcaraz_servizi.mp4

    # Video già importato: riprende da dove serve
    python src/run_pipeline.py --clip alcaraz_servizi

    # Dopo aver cambiato una soglia in config.yaml: rifà da uno step in poi
    python src/run_pipeline.py --clip alcaraz_servizi --from detect
"""

import argparse
import os
import shutil
import subprocess
import sys
from glob import glob

import cv2
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIR = os.path.join(ROOT, "clips")

CLIP_CONFIG_TEMPLATE = """\
# Parametri specifici per questa clip: sovrascrivono quelli di config.yaml
# nella cartella principale. Scrivi solo le voci da cambiare, mantenendo
# la stessa struttura. Esempio:
#
# detection:
#   peak_height: 0.25
#
# Gli fps vengono letti automaticamente dal video: non serve indicarli.
"""


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Unisce override dentro base, ricorsivamente sui sotto-dizionari."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        sys.exit(f"❌ Impossibile leggere gli fps da: {video_path}")
    return round(fps, 3)


def find_clip_video(clip_dir: str) -> str | None:
    videos = glob(os.path.join(clip_dir, "video.*"))
    return videos[0] if videos else None


def import_video(video_path: str, name: str | None) -> str:
    """Crea clips/<nome>/ e ci copia il video. Restituisce la cartella."""
    if not os.path.isfile(video_path):
        sys.exit(f"❌ Video non trovato: {video_path}")

    name = name or os.path.splitext(os.path.basename(video_path))[0]
    clip_dir = os.path.join(CLIPS_DIR, name)
    existing = find_clip_video(clip_dir)

    if existing:
        print(f"ℹ️  La clip '{name}' esiste già: uso il video in {existing}")
    else:
        os.makedirs(clip_dir, exist_ok=True)
        ext = os.path.splitext(video_path)[1].lower()
        dest = os.path.join(clip_dir, f"video{ext}")
        print(f"📁 Creo la clip '{name}' e copio il video in: {dest}")
        shutil.copy2(video_path, dest)

    return clip_dir


def write_effective_config(clip_dir: str, video_path: str, base_config_path: str) -> str:
    """
    Unisce config.yaml generale + config.yaml della clip + fps del video
    e salva il risultato in config_usato.yaml (passato a tutti gli script).
    """
    clip_config_path = os.path.join(clip_dir, "config.yaml")
    if not os.path.exists(clip_config_path):
        with open(clip_config_path, "w", encoding="utf-8") as f:
            f.write(CLIP_CONFIG_TEMPLATE)

    config = deep_merge(load_yaml(base_config_path), load_yaml(clip_config_path))
    config["statistics"]["fps"] = read_video_fps(video_path)

    effective_path = os.path.join(clip_dir, "config_usato.yaml")
    content = ("# Generato da run_pipeline.py - non modificare a mano.\n"
               "# Per cambiare un parametro solo per questa clip usa config.yaml.\n"
               + yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    with open(effective_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"⚙️  Config: {effective_path} (fps video = {config['statistics']['fps']})")
    return effective_path


def build_steps(clip_dir: str, video: str, config_path: str) -> list[dict]:
    """
    Definisce gli step della pipeline. Per ognuno:
      inputs   — file da cui dipende (se più recenti dell'output → si rifà)
      optional — input che, se presenti, contano come dipendenze
      output   — file prodotto (usato per capire se lo step è già fatto)
    """
    p = lambda *parts: os.path.join(clip_dir, *parts)
    keypoints, clean = p("keypoints.csv"), p("keypoints_clean.csv")
    shots, classified = p("shots.csv"), p("classified.csv")
    annotations = p("annotations.csv")
    has_ann = os.path.exists(annotations)
    ann_args = ["--annotations", annotations] if has_ann else []

    return [
        {"name": "extract", "script": "extract_keypoints.py",
         "args": ["--video", video, "--output", keypoints],
         "inputs": [video], "output": keypoints},
        {"name": "preprocess", "script": "preprocess.py",
         "args": ["--input", keypoints, "--output", clean],
         "inputs": [keypoints], "output": clean},
        {"name": "detect", "script": "detect_shots.py",
         "args": ["--keypoints", clean, "--output", shots] + ann_args,
         "inputs": [clean], "optional": [annotations], "output": shots},
        {"name": "classify", "script": "classify_shots.py",
         "args": ["--keypoints", clean, "--shots", shots, "--output", classified],
         "inputs": [clean, shots], "output": classified},
        {"name": "evaluate", "script": "evaluate.py",
         "args": ["--classified", classified, "--annotations", annotations,
                  "--output", p("results", "evaluation")],
         "inputs": [classified, annotations],
         "output": p("results", "evaluation", "classification_report.txt"),
         "requires_annotations": True},
        {"name": "stats", "script": "game_stats.py",
         "args": ["--classified", classified, "--output", p("results", "statistics")],
         "inputs": [classified], "output": p("results", "statistics", "statistics.json")},
        {"name": "plots", "script": "visualize.py",
         "args": ["--keypoints", clean, "--classified", classified,
                  "--output", p("results", "plots")],
         "inputs": [clean, classified], "output": p("results", "plots", "shot_timeline.png")},
        {"name": "render", "script": "render_video.py",
         "args": ["--video", video, "--keypoints", clean, "--classified", classified,
                  "--output", p("results", "annotated.mp4")],
         "inputs": [video, clean, classified], "output": p("results", "annotated.mp4")},
    ]


def is_up_to_date(step: dict) -> bool:
    output = step["output"]
    if not os.path.exists(output):
        return False
    deps = step["inputs"] + [f for f in step.get("optional", []) if os.path.exists(f)]
    out_time = os.path.getmtime(output)
    return all(os.path.exists(d) and os.path.getmtime(d) <= out_time for d in deps)


def run_step(step: dict, config_path: str):
    cmd = [sys.executable, os.path.join(ROOT, "src", step["script"])] + step["args"] \
          + ["--config", config_path]
    # PYTHONUTF8 evita errori di codifica delle emoji nel terminale di Windows
    env = dict(os.environ, PYTHONUTF8="1")
    sys.stdout.flush()  # stampa l'intestazione dello step prima dell'output del sottoprocesso
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        sys.exit(f"\n❌ Step '{step['name']}' fallito (codice {result.returncode}). Pipeline interrotta.")


def run_pipeline(clip_dir: str, base_config_path: str, from_step: str | None, force: bool):
    clip_name = os.path.basename(clip_dir)
    video = find_clip_video(clip_dir)
    if video is None:
        sys.exit(f"❌ Nessun file video.* in {clip_dir}")

    config_path = write_effective_config(clip_dir, video, base_config_path)
    steps = build_steps(clip_dir, video, config_path)
    step_names = [s["name"] for s in steps]
    force_from = 0 if force else (step_names.index(from_step) if from_step else None)

    for i, step in enumerate(steps):
        header = f"\n━━━ [{i + 1}/{len(steps)}] {step['name']} ━━━"

        if step.get("requires_annotations") and not os.path.exists(step["inputs"][-1]):
            print(f"{header}\n⏭️  Saltato: manca annotations.csv")
            continue

        forced = force_from is not None and i >= force_from
        if not forced and is_up_to_date(step):
            print(f"{header}\n✔️  Già aggiornato: {os.path.relpath(step['output'], ROOT)}")
            continue

        print(header)
        run_step(step, config_path)

    print(f"\n✅ Pipeline completata per la clip '{clip_name}'. Output in: "
          f"{os.path.relpath(os.path.join(clip_dir, 'results'), ROOT)}")

    if not os.path.exists(os.path.join(clip_dir, "annotations.csv")):
        print(f"\n📝 Manca {os.path.relpath(os.path.join(clip_dir, 'annotations.csv'), ROOT)}:"
              f" crealo guardando il video (colonne: frame,tipo) e rilancia con"
              f"\n   python src/run_pipeline.py --clip {clip_name}"
              f"\n   per ottenere la valutazione.")


def main():
    step_names = ["extract", "preprocess", "detect", "classify",
                  "evaluate", "stats", "plots", "render"]

    parser = argparse.ArgumentParser(description="Esegue la pipeline completa su una clip")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Nuovo video da importare in clips/<nome>/")
    source.add_argument("--clip", help="Nome di una clip già presente in clips/")
    parser.add_argument("--name", help="Nome della clip (default: nome del file video)")
    parser.add_argument("--from", dest="from_step", choices=step_names,
                        help="Rifà questo step e tutti i successivi")
    parser.add_argument("--force", action="store_true",
                        help="Rifà tutti gli step, compresa l'estrazione YOLO")
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"),
                        help="Config generale")
    args = parser.parse_args()

    if args.video:
        clip_dir = import_video(args.video, args.name)
    else:
        clip_dir = os.path.join(CLIPS_DIR, args.clip)
        if not os.path.isdir(clip_dir):
            sys.exit(f"❌ Clip non trovata: {clip_dir}")

    run_pipeline(clip_dir, args.config, args.from_step, args.force)


if __name__ == "__main__":
    main()
