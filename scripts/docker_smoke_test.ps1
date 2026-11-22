# Prueba de humo de la imagen de la Fase 2 (CLI): predice con data/sample_patients.csv dentro de Docker.
param(
    [string]$Image = "heart-disease-cli",
    [string]$InputFile = "data/sample_patients.csv",
    [string]$Output = "predicciones-docker-smoke.csv"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Construyendo imagen Docker..."
docker build -f fase-2/Dockerfile -t $Image .
if ($LASTEXITCODE -ne 0) { throw "docker build terminó con código $LASTEXITCODE" }

if (Test-Path $Output) { Remove-Item $Output -Force }

Write-Host "Ejecutando predicción dentro de Docker..."
docker run --rm -v "${PWD}:/work" $Image --data "/work/$InputFile" --output "/work/$Output"
if ($LASTEXITCODE -ne 0) { throw "Docker terminó con código de salida $LASTEXITCODE" }

if (-not (Test-Path $Output)) { throw "Docker no produjo el archivo $Output" }
$header = Get-Content $Output -TotalCount 1
if ($header -notmatch "prediction" -or $header -notmatch "probability") {
    throw "La salida no contiene prediction y probability"
}
Write-Host "Smoke test Docker exitoso"
