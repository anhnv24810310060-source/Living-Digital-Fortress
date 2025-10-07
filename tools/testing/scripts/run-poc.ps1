$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..')

Write-Host "Starting ShieldX-Cloud PoC services..." -ForegroundColor Cyan
Write-Host "Repo: $repoRoot"

$env:LOCATOR_PORT = $env:LOCATOR_PORT -as [string]
if (-not $env:LOCATOR_PORT) { $env:LOCATOR_PORT = "8080" }

$env:INGRESS_PORT = $env:INGRESS_PORT -as [string]
if (-not $env:INGRESS_PORT) { $env:INGRESS_PORT = "8081" }

$env:DECOY_PORT = $env:DECOY_PORT -as [string]
if (-not $env:DECOY_PORT) { $env:DECOY_PORT = "8082" }

# Hardened defaults
if (-not $env:WCH_MAX_ENVELOPE_BYTES) { $env:WCH_MAX_ENVELOPE_BYTES = "65536" }
if (-not $env:DECOY_JITTER_MS) { $env:DECOY_JITTER_MS = "120" }

$procs = @()

Push-Location $repoRoot

try {
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/locator" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/ingress" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/decoy-http" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/guardian" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/decoy-manager" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/shapeshifter" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/anchor" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/masque" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/sinkhole" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/decoy-ssh" -WorkingDirectory $repoRoot
  Start-Sleep -Milliseconds 300
  $procs += Start-Process -PassThru -NoNewWindow -FilePath "go" -ArgumentList "run ./services/decoy-redis" -WorkingDirectory $repoRoot

  Write-Host "Locator : http://localhost:$($env:LOCATOR_PORT)" -ForegroundColor Green
  Write-Host "Ingress : http://localhost:$($env:INGRESS_PORT)" -ForegroundColor Green
  Write-Host "Decoy   : http://localhost:$($env:DECOY_PORT)" -ForegroundColor Green
  Write-Host "Guardian: http://127.0.0.1:9090 (loopback only)" -ForegroundColor Green
  Write-Host "DecoyMgr: http://localhost:8083" -ForegroundColor Green
  Write-Host "Shifter : http://localhost:8084" -ForegroundColor Green
  Write-Host "Anchor  : http://localhost:8085" -ForegroundColor Green
  Write-Host "MASQUE  : QUIC :9444 (health :8086)" -ForegroundColor Green
  Write-Host "Sinkhole: TCP :9095, UDP :9096, HTTP :9097" -ForegroundColor Green
  Write-Host "DecoySSH: :2222" -ForegroundColor Green
  Write-Host "DecoyRED: :6380" -ForegroundColor Green
  Write-Host "Press Enter to stop all..." -ForegroundColor Yellow
  [void][System.Console]::ReadLine()
}
finally {
  foreach ($p in $procs) {
    try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
  }
  Pop-Location
}


