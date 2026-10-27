"""
evaluate.py — [Membro C]
==========================
Valuta le prestazioni della classificazione dei colpi confrontando
i risultati con le annotazioni manuali (ground truth).

Output:
  - classification_report.txt: colpi trovati, mancati e falsi, precisione
    e recall per tipo di colpo, elenco degli errori
  - errors.csv: colpi classificati male

Uso:
    python src/evaluate.py --classified clips/clip1/classified.csv
                           --annotations clips/clip1/annotations.csv
                           --output clips/clip1/results/evaluation/
"""

import argparse
import os

import numpy as np
import pandas as pd
import yaml


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
        pred = row.get("tipo_regole", "sconosciuto")

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


def build_report(matched: pd.DataFrame, n_annotated: int, tolerance_sec: float,
                 clip_name: str) -> str:
    """Report testuale in italiano: rilevamento, classificazione per tipo ed errori."""
    valid = matched[matched["true"].notna()]
    n_found = len(valid)
    n_false = len(matched) - n_found
    n_missed = n_annotated - n_found
    n_correct = int((valid["true"] == valid["pred"]).sum())
    pct = lambda num, den: f"{num / den * 100:.1f}%" if den else "-"

    title = f"VALUTAZIONE DEI COLPI - clip {clip_name}"
    lines = [title, "=" * len(title), "",
             f"1) RILEVAMENTO (un colpo è trovato se cade entro ±{tolerance_sec}s da quello annotato)",
             "",
             f"   Colpi annotati a mano ............ {n_annotated}",
             f"   Colpi rilevati dal sistema ....... {len(matched)}",
             f"   Trovati correttamente ............ {n_found}  ({pct(n_found, n_annotated)} degli annotati)",
             f"   Mancati (non rilevati) ........... {n_missed}",
             f"   Falsi colpi (rilevati in più) .... {n_false}",
             "",
             "2) CLASSIFICAZIONE (solo sui colpi trovati correttamente)",
             "",
             f"   Tipo giusto: {n_correct} su {n_found}  ({pct(n_correct, n_found)})",
             ""]

    header = f"   {'Tipo':<10}{'Annotati':>10}{'Predetti':>10}{'Corretti':>10}{'Precisione':>12}{'Recall':>9}"
    lines += [header, "   " + "-" * (len(header) - 3)]
    for tipo in sorted(set(valid["true"]) | set(valid["pred"])):
        n_true = int((valid["true"] == tipo).sum())
        n_pred = int((valid["pred"] == tipo).sum())
        ok = int(((valid["true"] == tipo) & (valid["pred"] == tipo)).sum())
        lines.append(f"   {tipo:<10}{n_true:>10}{n_pred:>10}{ok:>10}"
                     f"{pct(ok, n_pred):>12}{pct(ok, n_true):>9}")
    lines += ["",
              "   Annotati   = colpi di quel tipo secondo le annotazioni",
              "   Predetti   = colpi a cui il sistema ha dato quel tipo",
              "   Precisione = quando il sistema dice quel tipo, quante volte ha ragione (Corretti / Predetti)",
              "   Recall     = dei colpi veri di quel tipo, quanti vengono riconosciuti (Corretti / Annotati)",
              ""]

    errors = valid[valid["true"] != valid["pred"]]
    lines.append(f"3) ERRORI DI CLASSIFICAZIONE: {len(errors)}")
    if len(errors):
        lines += ["", f"   {'Frame':>7}{'Tempo (s)':>11}   {'Vero':<10}{'Predetto':<10}"]
        for _, row in errors.iterrows():
            lines.append(f"   {int(row['frame']):>7}{row['time_s']:>11.2f}   {row['true']:<10}{row['pred']:<10}")
    return "\n".join(lines) + "\n"


def evaluate(classified_path: str, annotations_path: str,
             output_dir: str, config: dict):
    cfg = config["detection"]

    print(f"📂 Carico classificazione: {classified_path}")
    classified = pd.read_csv(classified_path)

    print(f"📂 Carico annotazioni: {annotations_path}")
    annotations = pd.read_csv(annotations_path)

    print("🔗 Confronto colpi rilevati con annotazioni manuali...")
    tolerance = round(cfg["tolerance_sec"] * config["statistics"]["fps"])
    matched = match_shots_to_annotations(classified, annotations, tolerance)

    # Filtra solo i colpi con ground truth (escludi FP puri)
    matched_valid = matched[matched["true"].notna()].copy()
    print(f"   Colpi matchati: {len(matched_valid)} / {len(annotations)}")

    if len(matched_valid) == 0:
        print("❌ Nessun colpo matchato — verifica i file di input.")
        return

    clip_name = os.path.basename(os.path.dirname(os.path.abspath(classified_path)))
    report = build_report(matched, len(annotations), cfg["tolerance_sec"], clip_name)
    print("\n" + report)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Salva esempi di errori
    errors = matched_valid[matched_valid["true"] != matched_valid["pred"]]
    error_path = os.path.join(output_dir, "errors.csv")
    if len(errors) > 0:
        errors.to_csv(error_path, index=False)
        print(f"⚠️  Errori salvati in: {error_path}")
    elif os.path.exists(error_path):
        os.remove(error_path)  # errori di un'esecuzione precedente

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
