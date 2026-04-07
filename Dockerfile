# Étape 1 : Builder avec uv
FROM python:3.11-slim AS builder

# Installer uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copier les fichiers de configuration (pour profiter du cache Docker)
COPY .python-version pyproject.toml uv.lock* ./

# Installer les dépendances dans un venv
    RUN uv sync

# Étape 2 : Runtime
FROM python:3.11-slim

# Installer curl et uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Créer un user non-root pour la sécurité
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copier l'environnement virtuel depuis le builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copier le code de l'application
COPY --chown=appuser:appuser app/ ./app/

# Copier le dossier model et données de référence
COPY --chown=appuser:appuser model/ ./model/
COPY --chown=appuser:appuser data/clients_reference.parquet ./data/
COPY --chown=appuser:appuser data/drift_baseline.parquet ./data/

# Créer le dossier logs
RUN mkdir -p logs && chown appuser:appuser logs

# Passer à l'utilisateur non-root
USER appuser

# Configuration
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SCORE_THRESHOLD=0.5

# Exposer le port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de démarrage
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
