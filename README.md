# Credit Scoring API

API de scoring crédit avec monitoring et déploiement MLOps pour évaluer le risque de crédit des clients.

## 📋 Architecture

- **ML Model**: Pipeline sklearn avec LightGBM (preprocess + classifier)
- **API**: FastAPI avec client lookup basé sur SK_ID_CURR
- **Data**: Référence parquet avec 1000 clients + baseline drift avec 10000 clients
- **Container**: Docker multi-stage pour production
- **CI/CD**: GitHub Actions avec tests et déploiement

## 🚀 Démarrage rapide

### 1. Préparation des données

```bash
# Construire les datasets de référence et baseline
uv run python scripts/build_reference.py
```

Cela crée :
- `data/clients_reference.parquet` (1000 clients pour les prédictions)
- `data/drift_baseline.parquet` (10000 clients pour le monitoring drift)

### 2. Développement local

```bash
# Installer les dépendances
uv sync --all-extras

# Lancer l'API
uv run uvicorn app.main:app --reload

# Ou avec hot reload
uvicorn app.main:app --reload
```

L'API sera accessible sur http://localhost:8000

**Documentation interactive** : http://localhost:8000/docs

### 3. Tests

```bash
# Lancer les tests unitaires
uv run pytest tests/ -v

# Avec coverage
uv run pytest tests/ --cov=app
```

### 4. Production avec Docker

```bash
# Construire l'image
docker build -t credit-scoring-api:0.1.0 .

# Lancer le container
docker run -p 8000:8000 \
  -e SCORE_THRESHOLD=0.5 \
  credit-scoring-api:0.1.0

# Vérifier la santé
curl http://localhost:8000/health
```

## 📡 Endpoints

### Health Check
```bash
GET /health
```

### Prédiction de scoring crédit
```bash
POST /predict
Content-Type: application/json

{
  "client_id": 100001
}
```

**Réponse** :
```json
{
  "client_id": 100001,
  "score": 0.342,
  "prediction": 0,
  "threshold": 0.5,
  "inference_time_ms": 12.5
}
```

### Télécharger des client IDs valides pour les tests
```bash
GET /clients/sample?count=10
```

Retourne 10 client IDs aléatoires valides.

## 📊 Flux de données

```
data/application_train.csv (300k rows)
    ↓
scripts/build_reference.py
    ├→ data/clients_reference.parquet (1000 clients, indexé par SK_ID_CURR)
    └→ data/drift_baseline.parquet (10000 clients, pour monitoring)

POST /predict {client_id: 100001}
    ↓
app/main.py
    ↓
app/model.py (ModelService)
    ├→ Lookup client dans clients_reference.parquet
    ├→ Récupère 146 features
    ├→ Inférence via model/model.pkl
    └→ Return score + prediction

Logs → logs/api_logs.jsonl (format JSON)
```

## 🔧 Configuration

**Variable d'environnement** :
- `SCORE_THRESHOLD` (défaut : 0.5) - Seuil pour classification 0/1

**Fichiers de configuration** :
- `.python-version` - Version Python (3.11)
- `pyproject.toml` - Dépendances et métadonnées du projet

## 📦 Dépendances

| Package | Version | Rôle |
|---------|---------|------|
| fastapi | >=0.104.0 | Framework web |
| uvicorn | >=0.24.0 | ASGI server |
| scikit-learn | >=1.8.0 | ML pipeline |
| lightgbm | >=4.6.0 | Modèle de classificiation |
| pandas | >=3.0.2 | Data processing |
| pyarrow | >=18.0.0 | Parquet I/O |
| joblib | >=1.5.0 | Model serialization |

## 🧪 Tests

Les tests couvrent :
- Endpoints (health, predict, sample clients)
- Validation des inputs (format, types)
- Logique métier (cohérence score/threshold)
- Performance (< 500ms par requête)
- Gestion d'erreurs (client non trouvé, données invalides)

## 🐳 Docker

**Image de base** : `python:3.11-slim`

**Multi-stage build** :
1. Builder : Install dependencies
2. Runtime : Minimal image with model + reference data

**Optimisations** :
- Cache layers bien ordonnées
- User non-root (`appuser`)
- Health check intégré

## 🔄 CI/CD (GitHub Actions)

Pipeline automatisé sur push/PR vers `main` :

1. **Test** : Lancer les tests unitaires
2. **Build** : Construire et pusher l'image Docker
3. **Deploy** : Placeholder pour futur déploiement (Hugging Face Spaces, Cloud Run)

## 📝 Logging

Les prédictions sont loggées en JSON structuré dans `logs/api_logs.jsonl` :

```json
{
  "timestamp": "2026-04-07 10:30:45",
  "client_id": 100001,
  "score": 0.342,
  "prediction": 0,
  "threshold": 0.5,
  "inference_time_ms": 12.5,
  "features": {...}
}
```

## 🚨 Monitoring et Drift

Le fichier `data/drift_baseline.parquet` contient un échantillon baseline (10000 clients) sans SK_ID_CURR, destiné au monitoring du data drift. À implémenter dans `notebooks/monitoring.ipynb`.

## � Simulation de Trafic

Pour générer du trafic réaliste et remplir les logs à des fins de monitoring et d'analyse :

```bash
# Simulation basique (500 requêtes, sélection uniforme)
uv run python scripts/simulate_traffic.py

# Avec simulation de data drift (80% clients haut crédit, 20% random)
uv run python scripts/simulate_traffic.py --n-requests 1000 --drift

# Avec délai personnalisé (API distante)
uv run python scripts/simulate_traffic.py --url https://api.example.com --delay 0.1
```

**Options disponibles** :
- `--url` : URL de base de l'API (défaut: http://localhost:8000)
- `--n-requests` : Nombre de requêtes (défaut: 500)
- `--delay` : Délai entre requêtes en secondes (défaut: 0.05)
- `--drift` : Active la simulation de data drift

Voir [scripts/TRAFFIC_SIMULATION.md](scripts/TRAFFIC_SIMULATION.md) pour la documentation complète.

## 📖 Documentation additionnelle

- **Schemas** : [app/schemas.py](app/schemas.py) - Modèles Pydantic
- **Model Service** : [app/model.py](app/model.py) - Logique de prédiction
- **API** : [app/main.py](app/main.py) - Endpoints et lifecycle
- **Simulation de Trafic** : [scripts/TRAFFIC_SIMULATION.md](scripts/TRAFFIC_SIMULATION.md) - Load testing et drift simulation

---

**Version** : 0.1.0  
**Python** : >=3.11  
**Data Manager** : uv
