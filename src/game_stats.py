"""
game_stats.py — [Membro C]
============================
Calcola statistiche di gioco dai colpi classificati e le salva in CSV/JSON.

Statistiche prodotte:
  - Conteggio colpi per tipo
  - Tempo medio tra due colpi consecutivi
  - Percentuale dritto/rovescio
  - Colpi a rete e da fondo campo (stessa soglia su feet_advance usata
    dal classificatore per smash e volee)
  - Rovesci a una e a due mani (colonna two_hands_frac)

Uso:
    python src/game_stats.py --classified clips/clip1/classified.csv
                              --output clips/clip1/results/statistics/
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

    df["tipo"] = df["tipo_regole"]

    # Durata totale del clip
    durata_s = df["time_s"].max() - df["time_s"].min()

    # Conteggio colpi per tipo
    counts = df["tipo"].value_counts().to_dict()
    total = len(df)

    # Tempo medio tra colpi consecutivi
    df_sorted = df.sort_values("time_s")
    intervals = df_sorted["time_s"].diff().dropna()
    avg_interval = round(intervals.mean(), 3)

    # Percentuale dritto/rovescio
    n_dritto = counts.get("dritto", 0)
    n_rovescio = counts.get("rovescio", 0)
    pct_dritto = round(n_dritto / total * 100, 1) if total > 0 else 0
    pct_rovescio = round(n_rovescio / total * 100, 1) if total > 0 else 0

    # Colpi a rete: piedi più avanti della soglia del classificatore
    net_threshold = config["classification"]["rules"]["net_feet_advance_threshold"]
    n_rete = int((df["feet_advance"] > net_threshold).sum())
    pct_rete = round(n_rete / total * 100, 1) if total > 0 else 0

    # Rovesci a due mani: polsi vicini per buona parte della preparazione
    rovesci = df[df["tipo"] == "rovescio"]
    n_due_mani = int((rovesci["two_hands_frac"] >= config["statistics"]["two_hands_min_frac"]).sum())
    n_una_mano = len(rovesci) - n_due_mani
    pct_due_mani = round(n_due_mani / len(rovesci) * 100, 1) if len(rovesci) > 0 else 0

    stats = {
        "clip": os.path.basename(os.path.dirname(os.path.abspath(classified_path))),
        "durata_secondi": round(durata_s, 1),
        "totale_colpi": total,
        "colpi_per_tipo": counts,
        "tempo_medio_tra_colpi_s": avg_interval,
        "percentuale_dritto": pct_dritto,
        "percentuale_rovescio": pct_rovescio,
        "colpi_a_rete": n_rete,
        "colpi_da_fondo": total - n_rete,
        "percentuale_a_rete": pct_rete,
        "rovesci_a_due_mani": n_due_mani,
        "rovesci_a_una_mano": n_una_mano,
        "percentuale_rovesci_a_due_mani": pct_due_mani,
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
        print(f"   {tipo:12s}: {n:3d}")
    print(f"   Intervallo medio tra colpi: {avg_interval:.2f}s")
    print(f"   Dritto: {pct_dritto}% | Rovescio: {pct_rovescio}%")
    print(f"   A rete: {n_rete} ({pct_rete}%) | Da fondo: {total - n_rete}")
    print(f"   Rovesci a due mani: {n_due_mani} | a una mano: {n_una_mano}")

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
