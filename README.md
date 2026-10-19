# 🎾 Tennis Pose Analysis — Analisi Colpi di Tennis con YOLO

Progetto di **Image Processing** — Analisi automatica dei colpi di tennis da video di match (ripresa fissa da dietro il giocatore).

## Obiettivo

Il sistema analizza video di Sinner (ripresa da dietro) per:
- Estrarre 17 keypoint del corpo per ogni frame tramite **YOLO-Pose**
- Rilevare automaticamente i colpi
- Classificarli (dritto, rovescio, servizio, smash, volee) con regole if/else
- Produrre statistiche, grafici e video annotato

## Struttura del progetto

```
tennis-pose/
├── README.md
├── requirements.txt
├── config.yaml                  # soglie e parametri
├── src/
│   ├── run_pipeline.py          # esegue tutta la pipeline su una clip
│   ├── extract_keypoints.py     # [A] video → keypoints.csv (richiede GPU)
│   ├── preprocess.py            # [A] pulizia e normalizzazione del CSV
│   ├── detect_shots.py          # [B] rilevamento istanti dei colpi
│   ├── classify_shots.py        # [C] classificazione tipo di colpo
│   ├── evaluate.py              # [C] confronto con le annotazioni
│   ├── game_stats.py            # [C] statistiche finali
│   ├── visualize.py             # [C] grafici
│   └── render_video.py          # [A/C] video con scheletro ed etichette
├── notebooks/
│   ├── 01_explore_keypoints.ipynb
│   ├── 02_shot_detection.ipynb
│   └── 03_classification.ipynb
└── clips/                       # NON su GitHub (vedi .gitignore)
    └── <nome_clip>/             # una cartella per ogni video
        ├── video.mov            # copia del video originale
        ├── annotations.csv      # colpi annotati a mano (frame,tipo)
        ├── config.yaml          # parametri specifici della clip (opzionale)
        ├── config_usato.yaml    # config effettivo usato (generato)
        ├── keypoints.csv        # CSV grezzo di YOLO
        ├── keypoints_clean.csv  # CSV pulito e normalizzato
        ├── shots.csv            # colpi rilevati
        ├── classified.csv       # colpi classificati
        └── results/             # evaluation/, statistics/, plots/, annotated.mp4
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
yolo pose predict model=yolov8l-pose.pt source=clips/<nome_clip>/video.mov device=0
```

## Pipeline di elaborazione

```
video.mov
    ↓ [Membro A] extract_keypoints.py (richiede GPU)
keypoints.csv
    ↓ [Membro A] preprocess.py
keypoints_clean.csv
    ↓ [Membro B] detect_shots.py
shots.csv
    ↓ [Membro C] classify_shots.py
classified.csv
    ↓ [Membro C] evaluate.py + game_stats.py + visualize.py + render_video.py
results/
```

Tutti i file stanno in `clips/<nome_clip>/`. Non serve lanciare gli script uno per uno: lo fa `run_pipeline.py`.

### Aggiungere un nuovo video
```bash
python src/run_pipeline.py --video C:/percorso/alcaraz_servizi.mp4
```
Crea `clips/alcaraz_servizi/`, ci copia il video, legge gli fps dal video ed esegue tutti gli step.
Senza `annotations.csv` viene saltata la valutazione.

### Aggiungere le annotazioni
Crea `clips/<nome_clip>/annotations.csv` guardando il video (colonne `frame,tipo`), poi:
```bash
python src/run_pipeline.py --clip alcaraz_servizi
```
Vengono rifatti solo gli step il cui output è più vecchio dei suoi input: YOLO non viene rilanciato.

### Dopo aver cambiato una soglia
Le modifiche a `config.yaml` non vengono rilevate in automatico: indica da quale step ripartire.
```bash
python src/run_pipeline.py --clip alcaraz_servizi --from detect
```
Step disponibili: `extract`, `preprocess`, `detect`, `classify`, `evaluate`, `stats`, `plots`, `render`.
Con `--force` si rifà tutto, compresa l'estrazione YOLO.

Per cambiare un parametro **solo per una clip**, scrivilo in `clips/<nome_clip>/config.yaml`
(solo le voci da sovrascrivere, con la stessa struttura del `config.yaml` principale).

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
