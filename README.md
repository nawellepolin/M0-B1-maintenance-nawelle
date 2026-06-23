# M0-B1 — Service de maintenance prédictive (FastAPI + scikit-learn)

API REST de classification de la criticité d'incidents machine, exposant un modèle RandomForest pré-entraîné via FastAPI.

---

## Architecture

```
M0-B1-maintenance-nawelle/
├── app/
│   ├── main.py          # FastAPI : routes /health et /predict, lifespan, logging
│   └── schemas.py       # Pydantic : MachineInput, PredictionResponse, HealthResponse
├── data/
│   ├── generate_dataset.py   # génération du dataset synthétique (6 500 lignes)
│   └── maintenance_data.csv  # dataset d'entraînement
├── model/
│   ├── train_baseline.py     # pipeline sklearn : StandardScaler + OHE + RandomForest
│   └── model.joblib          # modèle sérialisé (~3 Mo compressé)
├── tests/
│   ├── test_health.py        # tests GET /health
│   ├── test_predict.py       # tests POST /predict (cas valide, erreur 422, parametrize)
│   └── test_addition.py      # test unitaire exemple
├── logs/
│   └── api.log          # journaux Loguru (rotation 5 MB, rétention 7 jours)
├── Dockerfile           # image python:3.11-slim + healthcheck
├── .dockerignore
└── requirements.txt
```

### Stack technique

| Composant | Technologie |
|---|---|
| API REST | FastAPI 0.115+ / Uvicorn |
| Validation | Pydantic v2 |
| ML | scikit-learn 1.9 — RandomForestClassifier (200 arbres) |
| Journalisation | Loguru (fichier + console) |
| Tests | Pytest + HTTPX (TestClient) |
| Conteneurisation | Docker (python:3.11-slim) |

### Flux d'une requête `/predict`

```
Client → POST /predict (JSON) → MachineInput (Pydantic)
       → DataFrame pandas → model.predict() + predict_proba()
       → PredictionResponse (JSON) → Client
                   ↕
             logs/api.log (Loguru)
```

Le modèle est chargé **une seule fois** au démarrage via le `lifespan` FastAPI et réutilisé pour toutes les requêtes.

---

## Installation

### Prérequis

- Python 3.11+
- (optionnel) Docker 24+

### Environnement local

```bash
# 1. Cloner le repo
git clone git@github.com:nawellepolin/M0-B1-maintenance-nawelle.git
cd M0-B1-maintenance-nawelle

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Lancement

### Local (développement)

```bash
uvicorn app.main:app --reload
```

L'API est disponible sur <http://localhost:8000>.  
La documentation Swagger interactive : <http://localhost:8000/docs>

### Docker (production)

```bash
# Build de l'image
docker build -t fastia-maintenance:dev .

# Lancement du conteneur
docker run --rm -p 8000:8000 fastia-maintenance:dev
```

Le conteneur expose le port `8000` et intègre un `HEALTHCHECK` sur `/health`.

---

## Endpoints

### `GET /health`

Vérifie que le service est opérationnel et que le modèle est chargé.

```bash
curl http://localhost:8000/health
```

Réponse attendue :

```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`

Prédit la criticité d'une machine à partir de 7 caractéristiques.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type_machine": "compresseur",
    "age_machine_jours": 1500,
    "derniere_maintenance_jours": 45,
    "temperature_moyenne": 68.5,
    "vibration_moyenne": 3.2,
    "pression_moyenne": 7.8,
    "nb_incidents_3_mois": 2
  }'
```

Réponse :

```json
{
  "criticite": "moyenne",
  "probabilites": {
    "basse": 0.12,
    "moyenne": 0.73,
    "haute": 0.15
  }
}
```

**Classes de criticité** : `basse` / `moyenne` / `haute`  
**Types de machine acceptés** : `pompe`, `compresseur`, `convoyeur`, `presse`, `four`

---

## Tests

```bash
# Lancer toute la suite
pytest

# Avec affichage détaillé
pytest -v

# Couverture par fichier
pytest -v tests/
```

La suite contient **7 tests** répartis en 3 fichiers :

| Fichier | Tests |
|---|---|
| `test_health.py` | Statut 200, schéma de réponse |
| `test_predict.py` | Cas valide, type invalide (422), 4 types de machine (parametrize) |
| `test_addition.py` | Test unitaire de base |

---

## Journaux

Les logs sont écrits dans `logs/api.log` (rotation à 5 MB, rétention 7 jours, compression zip).

Chaque requête `/predict` journalise :
- l'entrée reçue
- la classe prédite
- le temps de réponse en ms

```bash
# Suivre les logs en temps réel
tail -f logs/api.log
```

---

## Dépannage

| Symptôme | Solution |
|---|---|
| `Modèle introuvable` au démarrage | `python model/train_baseline.py` pour régénérer `model.joblib` |
| `ModuleNotFoundError` | Vérifier que le venv est activé et `pip install -r requirements.txt` |
| Port 8000 déjà utilisé | `uvicorn app.main:app --reload --port 8001` |
| `422 Unprocessable Entity` | Vérifier les valeurs envoyées (types, bornes Pydantic) |
