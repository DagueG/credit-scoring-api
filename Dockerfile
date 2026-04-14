# Étape 1 : Builder avec uv
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv
WORKDIR /app

COPY .python-version pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# Étape 2 : Runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copier le venv depuis le builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copier le code de l'application
COPY --chown=appuser:appuser app/ ./app/

# Copier le modèle et les données de référence (matérialisés via Git LFS)
COPY --chown=appuser:appuser model/ ./model/
COPY --chown=appuser:appuser data/clients_reference.parquet ./data/
COPY --chown=appuser:appuser data/drift_baseline.parquet ./data/

# Créer le dossier logs
RUN mkdir -p logs && chown appuser:appuser logs

USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SCORE_THRESHOLD=0.5

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
