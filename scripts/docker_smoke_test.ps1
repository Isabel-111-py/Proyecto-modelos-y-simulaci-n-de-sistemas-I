param(
    [string]$Image = "heart-failure-model",
    [string]$Model = "models/v3/model.pkl",
    [string]$InputFile = "pacientes.csv",
    [string]$Output = "predicciones-docker-smoke-latest.csv"
)

$ErrorActionPreference = "Stop"

Write-Host "Construyendo imagen Docker..."
docker build -t $Image .

Write-Host "Ejecutando predicción dentro de Docker..."

if (Test-Path $Output) {
    Remove-Item $Output -Force
}

docker run --rm `
  -v "${PWD}:/app" `
  $Image `
  --data $InputFile `
  --model $Model `
  --output $Output

if ($LASTEXITCODE -ne 0) {
    throw "Docker terminó con código de salida $LASTEXITCODE"
}

if (-not (Test-Path $Output)) {
    throw "Docker no produjo el archivo $Output"
}

$header = Get-Content $Output -TotalCount 1

if ($header -notmatch "prediction" -or
    $header -notmatch "probability") {
    throw "La salida no contiene prediction y probability"
}

Write-Host "Smoke test Docker exitoso"
