"""
compare_clips.py
================
Confronta le statistiche di tutte le clip già analizzate: legge
clips/<clip>/results/statistics/statistics.json di ogni clip e produce

    clips/confronto/
    ├── confronto.csv   # una riga per clip
    └── confronto.png   # tipi di colpo, colpi a rete, rovesci a due mani

Da lanciare dopo run_pipeline.py sulle clip da confrontare.

Uso:
    python src/compare_clips.py
    python src/compare_clips.py --clips sinner_30s federer
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from visualize import COLOR_MAP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIR = os.path.join(ROOT, "clips")
OUTPUT_DIR = os.path.join(CLIPS_DIR, "confronto")
SHOT_TYPES = list(COLOR_MAP)


def load_stats(clip_names: list[str] | None) -> pd.DataFrame:
    """Una riga per clip con le statistiche di statistics.json."""
    rows = []
    for name in sorted(clip_names or os.listdir(CLIPS_DIR)):
        path = os.path.join(CLIPS_DIR, name, "results", "statistics", "statistics.json")
        if not os.path.exists(path):
            if clip_names:
                print(f"⚠️  Statistiche mancanti per '{name}': lancia prima run_pipeline.py")
            continue
        with open(path, encoding="utf-8") as f:
            stats = json.load(f)
        row = {"clip": name}
        for key, value in stats.items():
            if key == "colpi_per_tipo":
                row.update({tipo: value.get(tipo, 0) for tipo in SHOT_TYPES})
            elif key != "clip":
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: str):
    """Tre grafici affiancati: tipi di colpo (%), colpi a rete (%), rovesci a due mani (%)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Tipi di colpo come percentuale del totale, barre impilate per clip
    bottom = pd.Series(0.0, index=df.index)
    for tipo in SHOT_TYPES:
        pct = df[tipo] / df["totale_colpi"] * 100
        if pct.sum() == 0:
            continue
        ax1.bar(df["clip"], pct, bottom=bottom, color=COLOR_MAP[tipo], label=tipo)
        bottom += pct
    ax1.set_ylim(0, 118)  # spazio per la legenda sopra le barre
    ax1.set_yticks(range(0, 101, 20))
    ax1.set_ylabel("% dei colpi")
    ax1.set_title("Tipi di colpo")
    ax1.legend(loc="upper center", ncol=5, fontsize=8, handlelength=1, columnspacing=0.8)

    for ax, col, title, color in [
        (ax2, "percentuale_a_rete", "Colpi giocati a rete", "#607D8B"),
        (ax3, "percentuale_rovesci_a_due_mani", "Rovesci a due mani", COLOR_MAP["rovescio"]),
    ]:
        ax.bar(df["clip"], df[col], color=color)
        for i, val in enumerate(df[col]):
            ax.text(i, val + 1.5, f"{val:.0f}%", ha="center", fontsize=10)
        ax.set_ylim(0, 110)
        ax.set_ylabel("%")
        ax.set_title(title)

    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis="x", rotation=20)

    path = os.path.join(output_dir, "confronto.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"   📊 Grafico di confronto salvato: {path}")


def main():
    parser = argparse.ArgumentParser(description="Confronta le statistiche delle clip analizzate")
    parser.add_argument("--clips", nargs="+", help="Clip da confrontare (default: tutte)")
    args = parser.parse_args()

    df = load_stats(args.clips)
    if df.empty:
        print("❌ Nessuna clip con statistiche trovata in clips/")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "confronto.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Tabella di confronto salvata: {csv_path}\n")
    print(df.set_index("clip").T.to_string())
    print()
    plot_comparison(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
