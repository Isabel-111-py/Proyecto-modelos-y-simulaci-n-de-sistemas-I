#!/usr/bin/env bash
# Prueba de humo de la imagen de la Fase 2 (CLI): predice con data/sample_patients.csv dentro de Docker.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="heart-disease-cli"
OUTPUT="predicciones-docker-smoke.csv"

docker build -f fase-2/Dockerfile -t "$IMAGE" .
rm -f "$OUTPUT"
docker run --rm -v "$PWD:/work" "$IMAGE" --data /work/data/sample_patients.csv --output "/work/$OUTPUT"

head -n 1 "$OUTPUT" | grep -q "prediction" && head -n 1 "$OUTPUT" | grep -q "probability"
echo "Smoke test Docker exitoso"
