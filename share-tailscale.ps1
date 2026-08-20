# Share the running AIC server with ONE trusted friend over Tailscale.
#
# This toggles HOST_BIND_ADDRESS to 0.0.0.0 (so devices on your tailnet can
# reach it), recreates the container, and prints the URL your friend opens.
#
# Prereqs:
#   1. BOTH you and your friend have the free Tailscale app installed and are
#      signed into the SAME account (tailscale.com), so you're on one tailnet.
#   2. Your friend must be a member of your tailnet.
#
# Then run:  .\share-tailscale.ps1
# Your friend opens:  http://<your-Tailscale-IP>:5000
#
# To take it back to local-only:  .\share-tailscale.ps1 -Off
[CmdletBinding()]
param(
    [switch]$Off   # switch back to localhost-only (127.0.0.1)
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path '.env')) {
    throw "Create $projectRoot\.env first (Copy-Item .env.example .env) and run run-docker.ps1 -Build once."
}

# Locate Tailscale IP
function Get-TailscaleIP {
    try {
        $json = tailscale ip -4 2>$null | Select-Object -First 1
        if ($json) { return $json.Trim() }
    } catch { }
    try {
        $json = & tailscale --socket "$env:USERPROFILE\AppData\Local\Tailscale\tailscaled.sock" ip -4 2>$null | Select-Object -First 1
        if ($json) { return $json.Trim() }
    } catch { }
    return $null
}

function Update-EnvBind([string]$addr) {
    $lines = Get-Content '.env'
    if ($lines -match '^HOST_BIND_ADDRESS=') {
        $lines = $lines | ForEach-Object { if ($_ -match '^HOST_BIND_ADDRESS=') { "HOST_BIND_ADDRESS=$addr" } else { $_ } }
    } else {
        $lines += "HOST_BIND_ADDRESS=$addr"
    }
    Set-Content '.env' $lines
}

if ($Off) {
    Update-EnvBind '127.0.0.1'
    docker compose --env-file .env up -d
    Write-Host 'Back to local-only: http://127.0.0.1:5000'
    exit 0
}

$ts = Get-TailscaleIP
if (-not $ts) {
    Write-Host "Warning: could not auto-detect your Tailscale IP." -ForegroundColor Yellow
    Write-Host 'Make sure Tailscale is installed and running. You can still continue.'
    $ts = Read-Host 'Enter your Tailscale IP (e.g. 100.101.102.103), or press Enter for 0.0.0.0'
    if (-not $ts) { $ts = '0.0.0.0' }
}

Update-EnvBind '0.0.0.0'
docker compose --env-file .env up -d

Write-Host ''
Write-Host 'Shares the AIC server on your Tailscale tailnet.'
if ($ts -and $ts -ne '0.0.0.0') {
    Write-Host "Your friend opens:  http://${ts}:5000"
} else {
    Write-Host 'Your friend opens:  http://<your-Tailscale-IP>:5000'
}
Write-Host 'CAUTION: 0.0.0.0 broadcasts on ALL interfaces. Keep this machine off'
Write-Host 'the open internet; this is only safe because Tailscale gives you a'
Write-Host 'private, encrypted network between just your devices.'
