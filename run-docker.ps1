[CmdletBinding()]
param(
    [switch]$Build,
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    throw "Created $projectRoot\.env. Edit its C:/example paths, then run this script again."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker Desktop is not available on PATH. Start/install Docker Desktop, then retry.'
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Desktop is not running or its Linux engine is unavailable.'
}

$compose = @('compose', '--env-file', '.env')
if ($Build) {
    & docker @compose build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($Foreground) {
    & docker @compose up
} else {
    & docker @compose up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'AIC is starting. Follow startup/model warmup with:'
        Write-Host '  docker compose --env-file .env logs -f retrieval-app'
        Write-Host 'Then open: http://127.0.0.1:5000'
    }
}
