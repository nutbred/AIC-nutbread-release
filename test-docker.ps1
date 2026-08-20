$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path '.env')) {
    throw 'Copy .env.example to .env and configure the data paths before testing.'
}

# Mount tests only for this validation command.
$testPath = (Join-Path $projectRoot 'tests').Replace('\', '/')
docker compose --env-file .env run --rm --no-deps `
    --volume "${testPath}:/tests:ro" `
    retrieval-app python -m unittest discover -s /tests -t /tests -v
