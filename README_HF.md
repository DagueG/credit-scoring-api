---
title: Credit Scoring API
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Credit Scoring API

API FastAPI de scoring crédit pour évaluer le risque de crédit des clients en temps réel. Utilise un modèle LightGBM entrainé avec preprocessing automatique et monitoring de drift des données via Evidently.

**Repository source**: https://github.com/yourusername/credit-scoring-api

## 🚀 Démarrage

Une fois le Space lancé, l'API sera accessible sur le port 7860.

## 📚 Documentation

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **Healthcheck**: `/health`

## 🔍 Endpoints

- `POST /predict` - Prédiction de scoring crédit pour un client
- `GET /health` - Vérification de l'état de l'API
- `GET/POST /feedback` - Dashboard de monitoring (Streamlit)

## 🧠 Modèle

- **Type**: Pipeline sklearn + LightGBM
- **Features**: 8 variables (EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, AMT_CREDIT, AMT_INCOME_TOTAL, AMT_ANNUITY, DAYS_BIRTH, DAYS_EMPLOYED)
- **Seuil de décision**: 0.5 (configurable via SCORE_THRESHOLD)

## 📊 Monitoring

L'API inclut un système de monitoring avec:
- Logs de toutes les prédictions en JSON
- Détection du drift via Evidently
- Dashboard Streamlit intégré

## 🛠️ Questions / Problèmes?

Créez une issue sur le [GitHub repository](https://github.com/yourusername/credit-scoring-api/issues)
