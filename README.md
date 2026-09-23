# 🎾 Tennis Pose Analysis — Analisi Colpi di Tennis con YOLO

Progetto di **Image Processing** — Analisi automatica dei colpi di tennis da video di match (ripresa fissa da dietro il giocatore).

## Obiettivo

Il sistema analizza video di Sinner (ripresa da dietro) per:
- Estrarre 17 keypoint del corpo per ogni frame tramite **YOLO-Pose**
- Rilevare automaticamente i colpi (dritto, rovescio, servizio)
- Classificarli con regole if/else o Random Forest
- Produrre statistiche, grafici e video annotato

## Struttura del progetto

```
tennis-pose/
├── README.md
├── requirements.txt
├── config.yaml                  # soglie e parametri
├── src/
│   ├── extract_keypoints.py     # [A] video → keypoints.csv (richiede GPU)
│   ├── preprocess.py            # [A] pulizia e normalizzazione del CSV
│   ├── detect_shots.py          # [B] rilevamento istanti dei colpi
│   ├── classify_shots.py        # [C] classificazione tipo di colpo
│   ├── evaluate.py              # [C] confronto con le annotazioni
│   ├── statistics.py            # [C] statistiche finali
│   ├── visualize.py             # [C] grafici
│   └── render_video.py          # [A/C] video con scheletro ed etichette
├── notebooks/
│   ├── 01_explore_keypoints.ipynb
│   ├── 02_shot_detection.ipynb
│   └── 03_classification.ipynb
├── data/                        # NON su GitHub (vedi .gitignore)
│   ├── raw/                     # video originali (.mp4)
│   ├── keypoints/               # CSV grezzi di YOLO
│   └── annotations/             # colpi annotati a mano
└── results/                     # output finale (grafici, video, statistiche)
```

## Setup rapido

### 1. Clonare il repository
```bash
git clone https://github.com/<tuo-username>/tennis-pose.git
cd tennis-pose
```

### 2. Creare ambiente virtuale e installare dipendenze
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. (Solo Membro A — PC con GPU NVIDIA) Setup YOLO con CUDA
```bash
# Verificare che CUDA sia disponibile
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Test rapido YOLO
yolo pose predict model=yolov8l-pose.pt source=data/raw/<clip>.mp4 device=0
```

## Pipeline di elaborazione

```
Video (.mp4)
    ↓ [Membro A] extract_keypoints.py (richiede GPU)
keypoints/<clip>.csv
    ↓ [Membro A] preprocess.py
keypoints_clean/<clip>.csv
    ↓ [Membro B] detect_shots.py
shots/<clip>_shots.csv
    ↓ [Membro C] classify_shots.py
shots/<clip>_classified.csv
    ↓ [Membro C] evaluate.py + statistics.py + visualize.py + render_video.py
results/
```

## Formato CSV keypoints (contratto tra i membri)

| Colonna | Significato |
|---------|-------------|
| `frame` | Numero del frame nel video |
| `time_s` | Secondo corrispondente |
| `track_id` | ID del giocatore dal tracker ByteTrack |
| `bbox_x1, bbox_y1, bbox_x2, bbox_y2` | Bounding box giocatore in pixel |
| `kp{0..16}_x, kp{0..16}_y, kp{0..16}_conf` | Coordinate e confidenza dei 17 keypoint COCO |

### Keypoint COCO più rilevanti

| ID | Parte | ID | Parte |
|----|-------|----|-------|
| 5 | Spalla sx | 6 | Spalla dx |
| 7 | Gomito sx | 8 | Gomito dx |
| 9 | Polso sx | **10** | **Polso dx** ← chiave rilevamento |
| 11 | Anca sx | 12 | Anca dx |

> **Nota su Sinner:** è mancino ma gioca il rovescio a due mani. La telecamera è da dietro, quindi la destra nell'immagine corrisponde alla destra reale — NON specchiare le regole.

## Team

| Membro | Ruolo | Fasi |
|--------|-------|------|
| A (GPU) | Infrastruttura, Estrazione keypoints, Preprocessing | 0, 2, 3 |
| B | Annotazione manuale, Rilevamento colpi | 1, 4 |
| C | Classificazione, Valutazione, Output finale | 5, 6, 7 |
