"""
evaluate.py — [Membro C]
==========================
Valuta le prestazioni della classificazione dei colpi confrontando
i risultati con le annotazioni manuali (ground truth).

Output:
  - Matrice di confusione (plot + testo)
  - Precision, Recall, F1 per tipo di colpo
  - Esempi di errori commentati

Uso:
    python src/evaluate.py --classified data/shots/clip1_classified.csv
                           --annotations data/annotations/clip1.csv
                           --output results/evaluation/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def match_shots_to_annotations(classified: pd.DataFrame,
                                annotations: pd.DataFrame,
                                tolerance: int) -> pd.DataFrame:
    """
    Per ogni colpo classificato, trova l'annotazione corrispondente
    entro la tolleranza in frame.
    """
    results = []
    used_gt = set()

    for _, row in classified.iterrows():
        frame = int(row["frame"])
        pred = row.get("tipo_rf", row.get("tipo_regole", "sconosciuto"))

        best_gt = None
        best_dist = tolerance + 1
        for idx, ann_row in annotations.iterrows():
            if idx in used_gt:
                continue
            dist = abs(frame - int(ann_row["frame"]))
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_gt = idx

        if best_gt is not None:
            used_gt.add(best_gt)
            true_label = annotations.loc[best_gt, "tipo"]
        else:
            true_label = None  # falso positivo

        results.append({
            "frame": frame,
            "time_s": row.get("time_s", np.nan),
            "pred": pred,
            "true": true_label,
        })

    return pd.DataFrame(results)


def plot_confusion_matrix(y_true, y_pred, labels, output_dir: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matrice di Confusione — Classificazione Colpi")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   📊 Matrice di confusione salvata: {path}")


def evaluate(classified_path: str, annotations_path: str,
             output_dir: str, config: dict):
    cfg = config["detection"]

    print(f"📂 Carico classificazione: {classified_path}")
    classified = pd.read_csv(classified_path)

    print(f"📂 Carico annotazioni: {annotations_path}")
    annotations = pd.read_csv(annotations_path)

    print("🔗 Confronto colpi rilevati con annotazioni manuali...")
    matched = match_shots_to_annotations(classified, annotations, cfg["tolerance_frames"])

    # Filtra solo i colpi con ground truth (escludi FP puri)
    matched_valid = matched[matched["true"].notna()].copy()
    print(f"   Colpi matchati: {len(matched_valid)} / {len(annotations)}")

    if len(matched_valid) == 0:
        print("❌ Nessun colpo matchato — verifica i file di input.")
        return

    labels = sorted(matched_valid["true"].unique())
    y_true = matched_valid["true"].tolist()
    y_pred = matched_valid["pred"].tolist()

    print("\n📊 Report di classificazione:")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)

    os.makedirs(output_dir, exist_ok=True)

    # Salva il report testuale
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Plot matrice di confusione
    plot_confusion_matrix(y_true, y_pred, labels, output_dir)

    # Salva esempi di errori
    errors = matched_valid[matched_valid["true"] != matched_valid["pred"]]
    if len(errors) > 0:
        error_path = os.path.join(output_dir, "errors.csv")
        errors.to_csv(error_path, index=False)
        print(f"\n⚠️  Errori trovati: {len(errors)} — salvati in: {error_path}")
        print(errors[["frame", "time_s", "true", "pred"]].head(10).to_string(index=False))
    else:
        print("\n✅ Nessun errore di classificazione trovato!")

    print(f"\n✅ Valutazione completata. Output in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Valuta la classificazione dei colpi")
    parser.add_argument("--classified", required=True, help="CSV colpi classificati")
    parser.add_argument("--annotations", required=True, help="CSV annotazioni manuali")
    parser.add_argument("--output", required=True, help="Cartella output risultati")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(args.classified, args.annotations, args.output, config)


if __name__ == "__main__":
    main()
