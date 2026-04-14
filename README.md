# Credit Scoring API — Déploiement MLOps

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/DagueGG/credit-scoring-api)
[![API Live](https://img.shields.io/badge/API%20Live-Available-success)](https://daguegg-credit-scoring-api.hf.space)
[![Swagger Docs](https://img.shields.io/badge/Swagger-API%20Docs-green)](https://daguegg-credit-scoring-api.hf.space/docs)

---

## 📖 Contexte

Projet **OpenClassrooms — Prêt à Dépenser**. Mise en production d'un modèle de scoring crédit (LightGBM) développé en phase initiale avec MLflow. Ce projet démontre l'implémentation complète de la chaîne MLOps : API production-ready, CI/CD automatisé, monitoring du data drift, et déploiement sur infrastructure cloud.

---

## 🏗️ Architecture du projet

### Flux de traitement

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client / Frontend                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /predict
                             ▼
        ┌─────────────────────────────────────────┐
        │         FastAPI REST Endpoint           │
        │  • Validation Pydantic                  │
        │  • Logging structuré (JSON)             │
        └────────────┬────────────────────────────┘
                     │ Lookup client_id
                     ▼
        ┌─────────────────────────────────────────┐
        │   Parquet Référence (1000 clients)      │
        │  Extraction des features                │
        └────────────┬────────────────────────────┘
                     │ Features → Model Input
                     ▼
        ┌─────────────────────────────────────────┐
        │   LightGBM Pipeline (sklearn)           │
        │  • Preprocessing + Feature Engineering  │
        │  • Classification binaire                │
        └────────────┬────────────────────────────┘
                     │ Prediction + Score
                     ▼
        ┌─────────────────────────────────────────┐
        │    Response + JSON Structuré            │
        │  • Score (0-1)                          │
        │  • Prédiction (0=Approuvé, 1=Rejeté)   │
        │  • Latence d'inférence                  │
        └────────────┬────────────────────────────┘
                     │ Log JSON → logs/api_logs.jsonl
                     ▼
        ┌─────────────────────────────────────────┐
        │  Monitoring & Détection de Drift        │
        │  • Evidently AI                         │
        │  • Dashboard Streamlit                  │
        │  • Rapports HTML (reports/)             │
        └─────────────────────────────────────────┘
```

### Structure du projet

```
.
├── app/                           # Code source de l'API
│   ├── __init__.py
│   ├── main.py                    # Application FastAPI, endpoints
│   ├── model.py                   # Service de chargement du modèle
│   └── schemas.py                 # Schémas Pydantic (CreditRequest, CreditResponse)
│
├── data/                          # Données (Git LFS)
│   ├── application_train.csv      # Source brute de la training (ref)
│   ├── clients_reference.parquet  # 1000 clients pour les prédictions en production
│   └── drift_baseline.parquet     # 10000 clients baseline pour la détection de drift
│
├── model/                         # Binaires du modèle (Git LFS)
│   └── credit_model.pkl           # Pipeline sklearn (preprocessing + LightGBM)
│
├── scripts/                       # Utilitaires et scripts batch
│   ├── build_reference.py         # Construction de clients_reference.parquet et drift_baseline.parquet
│   └── simulate_traffic.py        # Simulation de trafic pour tests et génération de rapports
│
├── tests/                         # Test suite automatisée (pytest)
│   ├── __init__.py
│   └── test_api.py                # Tests unitaires et d'intégration
│
├── notebooks/                     # Notebooks Jupyter
│   └── monitoring.ipynb           # Analyse du drift avec Evidently, génération de rapports HTML
│
├── dashboard/                     # Application Streamlit pour le monitoring
│   ├── __init__.py
│   └── app.py                     # Dashboard interactif
│
├── reports/                       # Rapports et sorties d'analyse
│   ├── drift_report.csv           # Métriques de drift
│   └── quality_report.csv         # Métriques de qualité
│
├── logs/                          # Logs d'exécution
│   └── api_logs.jsonl             # Logs structurés en JSON (1 prédiction = 1 ligne)
│
├── Dockerfile                     # Image Docker multi-stage pour production
├── pyproject.toml                 # Configuration du projet (dépendances, metadata)
├── .github/workflows/             # CI/CD GitHub Actions
├── README.md                      # Ce fichier
└── README_HF.md                   # Documentation spécifique à Hugging Face Spaces
```

---

## 🔧 Choix techniques

| Composant | Choix | Justification |
|-----------|-------|-------------|
| **Package Manager** | `uv` (au lieu de pip) | ⚡ Rapidité (~100x plus rapide), génération automatique de lockfile, reproductibilité garantie |
| **Framework API** | FastAPI (au lieu de Gradio) | 🚀 API REST standard, validation Pydantic, documentation Swagger auto, support async/concurrence, production-ready |
| **Lookup Features** | client_id (au lieu de 146 features en input) | 📊 Réaliste, léger, réutilise la référence pour le drift, simule un appel en production réelle |
| **Drift Detection** | Evidently AI (au lieu de custom) | 📈 Standard industriel, rapports HTML riches, intégration pandas native, benchmark statistiques robustes |
| **Dashboard** | Streamlit | 🎨 Développement rapide, composants prêts à l'emploi, idéal pour prototypage et monitoring |
| **Déploiement** | Hugging Face Spaces + Docker | 🌐 Gratuit, support Docker natif, CI/CD simple, accès URL public immédiat |
| **Versioning Binaires** | Git LFS | 🔐 Stockage efficace du modèle (.pkl) et des parquets, pas de pollution du dépôt Git |

---

## 📦 Prérequis d'installation

- **uv** ≥ 0.1.0 : [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **Python** ≥ 3.11
- **Docker** (optionnel, pour containerisation)
- **Git LFS** (pour récupérer modèle et données)

### Vérifier l'installation

```bash
uv --version
python --version
git lfs version
docker --version  # optionnel
```

---

## 🚀 Installation locale

### 1. Cloner et initialiser le dépôt

```bash
git clone https://github.com/DagueGG/credit-scoring-api.git
cd credit-scoring-api

# Télécharger les binaires (modèle, parquets via Git LFS)
git lfs pull
```

### 2. Installer les dépendances

```bash
# Installation complète avec dépendances de dev
uv sync --all-extras

# Ou sans dépendances de dev (production)
uv sync
```

### 3. Construire les données de référence (si absent)

Si les parquets `data/clients_reference.parquet` et `data/drift_baseline.parquet` ne sont pas présents :

```bash
uv run python scripts/build_reference.py
```

Cela génère :
- **clients_reference.parquet** : 1000 clients pour les prédictions
- **drift_baseline.parquet** : 10000 clients pour la baseline du monitoring

---

## 🏃 Lancer l'API en local

### Mode développement (hot reload)

```bash
uv run uvicorn app.main:app --reload
```

L'API démarrera sur **http://localhost:8000**

### Vérifier le statut

```bash
curl http://localhost:8000/health
```

Réponse attendue :
```json
{"status": "healthy"}
```

### Exemple de requête de prédiction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"client_id": 100001}'
```

Réponse :
```json
{
  "client_id": 100001,
  "score": 0.342,
  "prediction": 0,
  "threshold": 0.5,
  "inference_time_ms": 12.5
}
```

### Documentation interactive

Accédez à **http://localhost:8000/docs** pour explorer l'API avec Swagger UI.

---

## 🐳 Lancer avec Docker

### Construire l'image

```bash
docker build -t credit-scoring-api:latest .
```

### Lancer le conteneur

```bash
docker run -p 7860:7860 \
  --name credit-api \
  credit-scoring-api:latest
```

L'API sera accessible sur **http://localhost:7860**

### Arrêter le conteneur

```bash
docker stop credit-api
docker rm credit-api
```

---

## 📡 Endpoints de l'API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/health` | `GET` | Vérifier le statut de l'API et du modèle |
| `/predict` | `POST` | Générer une prédiction de scoring crédit pour un client |
| `/clients/sample` | `GET` | Récupérer un client exemple pour tester l'API |

### Détails des endpoints

#### `GET /health`
Retourne le statut de santé de l'API.

```json
{
  "status": "healthy"
}
```

#### `POST /predict`
Prédiction de scoring pour un client.

**Request Body :**
```json
{
  "client_id": 100001
}
```

**Response (200 OK) :**
```json
{
  "client_id": 100001,
  "score": 0.342,
  "prediction": 0,
  "threshold": 0.5,
  "inference_time_ms": 12.5
}
```

**Response (404 Not Found) :**
```json
{
  "detail": "Client 999999 not found in reference data"
}
```

#### `GET /clients/sample`
Retourne un ID de client valide pour les tests.

```json
{
  "client_id": 100001
}
```

---

## ✅ Tests

### Lancer la suite de tests

```bash
uv run pytest tests/ -v
```

### Lancer avec rapport de coverage

```bash
uv run pytest tests/ --cov=app --cov-report=html
```

Coverage généré dans `htmlcov/`, rapport consultable dans `htmlcov/index.html`

### Contenu des tests

Les tests couvrent :
- ✅ **Fonctionnels** : endpoints disponibles, statuts HTTP corrects
- ✅ **Validation** : schémas Pydantic, traitement des entrées invalides
- ✅ **Cohérence métier** : prédictions cohérentes, seuil appliqué correctement
- ✅ **Performance** : latence d'inférence < seuil acceptable
- ✅ **Données** : lookup client_id correct, features disponibles

---

## 🚗 Simulation de trafic

Script pour générer des requêtes et logs d'API, utile pour tester le monitoring.

### Lancer sans drift (trafic normal)

```bash
uv run python scripts/simulate_traffic.py --requests 100
```

Simule 100 prédictions avec des clients tirés aléatoirement de la référence.

### Lancer avec drift (trafic anomal)

```bash
uv run python scripts/simulate_traffic.py --requests 100 --drift
```

Simule 100 prédictions avec des features altérées (approx. stationnaires) pour générer un **drift détectable**. Utile pour tester la chaîne de monitoring.

### Résultat

Les logs sont écrits dans `logs/api_logs.jsonl`, 1 requête = 1 ligne JSON.

---

## 📊 Monitoring

### Notebook d'analyse (Evidently AI)

```bash
uv run jupyter notebook notebooks/monitoring.ipynb
```

Ce notebook :
1. **Charge les données** : clients_reference (baseline) et drift_baseline
2. **Compare les distributions** : détecte les dérives statistiques (Kolmogorov-Smirnov, etc.)
3. **Génère des rapports HTML** : sortie dans `reports/`
   - `drift_report.html` : analyses détaillées du drift par feature
   - `quality_report.html` : métriques de qualité du modèle

**Interprétation des rapports :**
- | **Top features en drift** | Features avec plus haute variation statistique (prob. causes = distribution changement) |
- | **Seuils de drift** | Seuil p-value standard (0.05) ; si p < 0.05, drift détecté |
- | **Drift explicatif** | Quand `--drift` utilisé, l'altération volontaire des features génère un pattern détectable |
- | **Latence & Throughput** | Métriques opérationnelles : latence moyenne inférence, taux prédictions/sec |

### Dashboard Streamlit

```bash
uv run streamlit run dashboard/app.py
```

Accès sur **http://localhost:8501**

Affiche :
- 📈 Historique des scores et prédictions
- 🔄 Indicateurs de drift (top features)
- 💾 Tendances des logs API
- ⚠️ Alertes si drift détecté

---

## 🔄 Pipeline CI/CD

Pipeline GitHub Actions : `.github/workflows/ci_cd.yml`

### Jobs exécutés à chaque `push` sur `main` :

| Job | Étapes | Sortie |
|-----|--------|--------|
| **test** | `uv sync → pytest` | Tests automatisés, couverts ou échouent |
| **build** | `Docker build → Tag` | Image Docker construite et tagguée |
| **deploy** | `Docker push → Docker Hub` | Image pushée à `docker.io/daguegg/credit-scoring-api:latest` |
| **deploy-huggingface** | `Push vers HF Spaces repo` | Déploiement auto sur HF Spaces |

### Secrets utilisés

| Secret | Utilité |
|--------|---------|
| `DOCKER_USERNAME` | Login Docker Hub |
| `DOCKER_PASSWORD` | Token Docker Hub |
| `HF_TOKEN` | Token Hugging Face (write access) |
| `HF_USERNAME` | Utilisateur Hugging Face |

### Déclencheur

Automatique à chaque `push` ou `merge` sur `main`.

---

## 🌐 Déploiement Hugging Face Spaces

### URL de production

**[https://daguegg-credit-scoring-api.hf.space](https://daguegg-credit-scoring-api.hf.space)**

### Configuration

- **Mode** : Docker
- **Port** : 7860
- **HF Token** : Configuré en secret de GitHub Actions
- **Déploiement** : Automatique après chaque push sur `main`

### Vérifier le déploiement

```bash
curl https://daguegg-credit-scoring-api.hf.space/health
curl -X POST https://daguegg-credit-scoring-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"client_id": 100001}'
```

---

## 📋 Livrables

- ✅ **API FastAPI fonctionnelle** : endpoints testés, validation Pydantic, documentation Swagger
- ✅ **Containerisation Docker** : Dockerfile multi-stage, image optimisée, déploiement robuste
- ✅ **Tests automatisés** : suite pytest couvrant fonctionnel, validation, métier, performance
- ✅ **Pipeline CI/CD** : GitHub Actions avec test → build → deploy → HF Spaces
- ✅ **Monitoring du data drift** : Evidently AI, rapports HTML, détection automatique
- ✅ **Dashboard Streamlit** : visualisation des logs, tendances, alertes
- ✅ **Déploiement en production** : HF Spaces + Docker, accessible 24/7
- ✅ **Documentation complète** : README détaillé, exemples curl, architecture expliquée

---

## 📚 Structure des logs API

Chaque prédiction génère une ligne JSON dans `logs/api_logs.jsonl` :

```json
{
  "timestamp": "2024-01-15T14:32:45.123456",
  "client_id": 100001,
  "score": 0.342,
  "prediction": 0,
  "threshold": 0.5,
  "inference_time_ms": 12.5,
  "status": "success"
}
```

Format exploitable directement par Evidently pour l'analyse du drift.

---

## 🔐 Sécurité et bonnes pratiques

- **Validation entrées** : Pydantic schemas + typage strict
- **Logging structuré** : JSON pour faciliter parsing et monitoring
- **Healthcheck Docker** : Détecte et redémarre conteneurs cassés
- **User non-root** : Exécution sous `appuser` dans Docker
- **Dependencies pinned** : uv.lock pour reproductibilité
- **Environment variables** : Configuration via variables (SCORE_THRESHOLD, etc.)

---

## 📞 Support

Pour toute question ou issue :
1. Consultez la documentation Swagger : https://daguegg-credit-scoring-api.hf.space/docs
2. Vérifiez les logs : `logs/api_logs.jsonl`
3. Lancez les tests localement : `uv run pytest tests/ -v`

---

**Dernière mise à jour** : avril 2026





