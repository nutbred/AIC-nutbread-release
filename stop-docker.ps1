$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path '.env')) {
    throw 'No .env exists in this project directory; nothing to stop.'
}

docker compose --env-file .env down
