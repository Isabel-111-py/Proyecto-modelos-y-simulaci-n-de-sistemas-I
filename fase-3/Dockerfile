# Fase 3 - API REST (FastAPI + Uvicorn).
# Construir desde la raíz del repositorio:  docker build -f fase-3/Dockerfile -t heart-disease-api:fase3 .
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# XGBoost necesita el runtime de OpenMP en las imágenes slim de Debian.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fase-2/*.py ./fase-2/
COPY fase-3/*.py ./fase-3/
COPY models/ ./models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "main:app", "--app-dir", "fase-3", "--host", "0.0.0.0", "--port", "8000"]
