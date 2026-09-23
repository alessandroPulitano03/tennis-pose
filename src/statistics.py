"""
statistics.py — [Membro C]
============================
Calcola statistiche di gioco dai colpi classificati e le salva in CSV/JSON.

Statistiche prodotte:
  - Conteggio colpi per tipo
  - Colpi al minuto per tipo
  - Tempo medio tra due colpi consecutivi
  - Percentuale dritto/rovescio

Uso:
    python src/statistics.py --classified data/shots/clip1_classified.csv
                              --output results/statistics/
"""

import argparse
import json
import os

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_statistics(classified_path: str, output_dir: str, config: dict):
    print(f"📂 Carico: {classified_path}")
    df = pd.read_csv(classified_path)

    # Usa tipo_rf se disponibile, altrimenti tipo_regole
    tipo_col = "tipo_rf" if "tipo_rf" in df.columns else "tipo_regole"
    df["tipo"] = df[tipo_col]

    # Durata totale del clip
    durata_s = df["time_s"].max() - df["time_s"].min()
    durata_min = durata_s / 60

    # Conteggio colpi per tipo
    counts = df["tipo"].value_counts().to_dict()
    total = len(df)

    # Colpi al minuto
    rate = {k: round(v / durata_min, 2) for k, v in counts.items()}

    # Tempo medio tra colpi consecutivi
    df_sorted = df.sort_values("time_s")
    intervals = df_sorted["time_s"].diff().dropna()
    avg_interval = round(intervals.mean(), 3)
    std_interval = round(intervals.std(), 3)

    # Percentuale dritto/rovescio
    n_dritto = counts.get("dritto", 0)
    n_rovescio = counts.get("rovescio", 0)
    pct_dritto = round(n_dritto / total * 100, 1) if total > 0 else 0
    pct_rovescio = round(n_rovescio / total * 100, 1) if total > 0 else 0

    stats = {
        "clip": os.path.basename(classified_path),
        "durata_secondi": round(durata_s, 1),
        "totale_colpi": total,
        "colpi_per_tipo": counts,
        "colpi_per_minuto": rate,
        "tempo_medio_tra_colpi_s": avg_interval,
        "deviazione_std_intervallo_s": std_interval,
        "percentuale_dritto": pct_dritto,
        "percentuale_rovescio": pct_rovescio,
    }

    os.makedirs(output_dir, exist_ok=True)

    # Salva JSON
    json_path = os.path.join(output_dir, "statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✅ Statistiche salvate in: {json_path}")

    # Stampa sommario
    print("\n📊 Sommario statistiche:")
    print(f"   Durata clip:     {stats['durata_secondi']:.1f}s")
    print(f"   Totale colpi:    {total}")
    for tipo, n in counts.items():
        print(f"   {tipo:12s}: {n:3d} ({rate[tipo]:.1f}/min)")
    print(f"   Intervallo medio tra colpi: {avg_interval:.2f}s ± {std_interval:.2f}s")
    print(f"   Dritto: {pct_dritto}% | Rovescio: {pct_rovescio}%")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Calcola statistiche dai colpi classificati")
    parser.add_argument("--classified", required=True, help="CSV colpi classificati")
    parser.add_argument("--output", required=True, help="Cartella output statistiche")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    compute_statistics(args.classified, args.output, config)


if __name__ == "__main__":
    main()
