"""
visualize.py — [Membro C]
===========================
Genera i grafici per la relazione finale:
  1. Velocità del polso nel tempo con i colpi evidenziati
  2. Distribuzione dei tipi di colpo (torta e barre)
  3. Boxplot delle misure cinematiche per tipo di colpo
  4. Linea temporale dei colpi

Uso:
    python src/visualize.py --keypoints data/keypoints/clip1_clean.csv
                             --classified data/shots/clip1_classified.csv
                             --output results/plots/
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Stile globale
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_MAP = {
    "dritto": "#2196F3",
    "rovescio": "#F44336",
    "servizio": "#4CAF50",
    "smash": "#FF9800",
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_wrist_speed(df_kp: pd.DataFrame, shots_df: pd.DataFrame,
                     output_dir: str):
    """Velocità del polso destro nel tempo con i colpi evidenziati."""
    speed = np.sqrt(df_kp["kp10_x"].diff() ** 2 + df_kp["kp10_y"].diff() ** 2)

    tipo_col = "tipo_rf" if "tipo_rf" in shots_df.columns else "tipo_regole"

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_kp["time_s"], speed, color="gray", linewidth=0.8, label="Velocità polso dx")

    for _, row in shots_df.iterrows():
        tipo = row.get(tipo_col, "dritto")
        color = COLOR_MAP.get(tipo, "black")
        ax.axvline(row["time_s"], color=color, alpha=0.7, linewidth=1.5)

    # Legenda colori
    patches = [mpatches.Patch(color=c, label=t) for t, c in COLOR_MAP.items()
               if t in shots_df[tipo_col].values]
    ax.legend(handles=patches, loc="upper right")
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Velocità normalizzata")
    ax.set_title("Velocità del polso destro con colpi evidenziati")

    path = os.path.join(output_dir, "wrist_speed.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"   📈 Velocità polso salvata: {path}")


def plot_shot_distribution(shots_df: pd.DataFrame, output_dir: str):
    """Distribuzione dei tipi di colpo: torta + barre."""
    tipo_col = "tipo_rf" if "tipo_rf" in shots_df.columns else "tipo_regole"
    counts = shots_df[tipo_col].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Torta
    colors = [COLOR_MAP.get(t, "gray") for t in counts.index]
    ax1.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90)
    ax1.set_title("Distribuzione colpi")

    # Barre
    ax2.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Numero di colpi")
    ax2.set_title("Conteggio per tipo di colpo")
    for i, (tipo, val) in enumerate(counts.items()):
        ax2.text(i, val + 0.3, str(val), ha="center", fontsize=10)

    path = os.path.join(output_dir, "shot_distribution.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"   📊 Distribuzione colpi salvata: {path}")


def plot_feature_boxplots(shots_df: pd.DataFrame, output_dir: str):
    """Boxplot delle feature cinematiche per tipo di colpo."""
    tipo_col = "tipo_rf" if "tipo_rf" in shots_df.columns else "tipo_regole"

    feature_cols = [c for c in ["wrist_r_x", "wrist_r_y", "swing_direction",
                                 "wrist_above_nose", "bimanual_dist"]
                    if c in shots_df.columns]
    if not feature_cols:
        print("   ⚠️  Feature non presenti nel CSV — salto boxplot.")
        return

    fig, axes = plt.subplots(1, len(feature_cols), figsize=(4 * len(feature_cols), 5))
    if len(feature_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, feature_cols):
        data = [shots_df[shots_df[tipo_col] == t][col].dropna().values
                for t in shots_df[tipo_col].unique()]
        labels = list(shots_df[tipo_col].unique())
        colors = [COLOR_MAP.get(t, "gray") for t in labels]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(col.replace("_", " "))
        ax.set_ylabel("Valore normalizzato")

    path = os.path.join(output_dir, "feature_boxplots.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"   📦 Boxplot feature salvato: {path}")


def plot_timeline(shots_df: pd.DataFrame, output_dir: str):
    """Linea temporale dei colpi."""
    tipo_col = "tipo_rf" if "tipo_rf" in shots_df.columns else "tipo_regole"

    fig, ax = plt.subplots(figsize=(14, 2))
    for _, row in shots_df.iterrows():
        tipo = row.get(tipo_col, "dritto")
        color = COLOR_MAP.get(tipo, "black")
        ax.scatter(row["time_s"], 0, color=color, s=80, zorder=3)

    patches = [mpatches.Patch(color=c, label=t) for t, c in COLOR_MAP.items()
               if t in shots_df[tipo_col].values]
    ax.legend(handles=patches, loc="upper right", ncol=4)
    ax.set_xlabel("Tempo (s)")
    ax.set_yticks([])
    ax.set_title("Timeline dei colpi")

    path = os.path.join(output_dir, "shot_timeline.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"   📅 Timeline salvata: {path}")


def visualize(keypoints_path: str, classified_path: str,
              output_dir: str, config: dict):
    print(f"📂 Carico keypoints: {keypoints_path}")
    df_kp = pd.read_csv(keypoints_path)

    print(f"📂 Carico colpi classificati: {classified_path}")
    shots_df = pd.read_csv(classified_path)

    os.makedirs(output_dir, exist_ok=True)

    print("📈 Genero grafici...")
    plot_wrist_speed(df_kp, shots_df, output_dir)
    plot_shot_distribution(shots_df, output_dir)
    plot_feature_boxplots(shots_df, output_dir)
    plot_timeline(shots_df, output_dir)

    print(f"\n✅ Tutti i grafici salvati in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Genera grafici per la relazione finale")
    parser.add_argument("--keypoints", required=True, help="CSV keypoints pulito")
    parser.add_argument("--classified", required=True, help="CSV colpi classificati")
    parser.add_argument("--output", required=True, help="Cartella output grafici")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    visualize(args.keypoints, args.classified, args.output, config)


if __name__ == "__main__":
    main()
