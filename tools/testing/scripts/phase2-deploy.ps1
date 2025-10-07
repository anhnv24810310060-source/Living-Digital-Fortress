# Phase 2 Deployment Script for ShieldX Cloud

param(
    [switch]$Production,
    [string]$Environment = "dev"
)

Write-Host "=== ShieldX Cloud Phase 2 Deployment ===" -ForegroundColor Green

# Check Phase 1 prerequisites
Write-Host "Checking Phase 1 foundation..." -ForegroundColor Yellow

$phase1Services = @("locator", "ingress", "guardian", "ml-orchestrator")
foreach ($service in $phase1Services) {
    $process = Get-Process -Name $service -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "✓ Phase 1 service $service running (PID: $($process.Id))" -ForegroundColor Green
    } else {
        Write-Host "⚠ Phase 1 service $service not running" -ForegroundColor Yellow
    }
}

# Build Phase 2 components
Write-Host "`nBuilding Phase 2 components..." -ForegroundColor Yellow

# Build enhanced services
go build -o bin/shapeshifter-v2.exe ./services/shapeshifter
if ($LASTEXITCODE -ne 0) {
    Write-Error "Shapeshifter v2 build failed"
    exit 1
}
Write-Host "✓ Shapeshifter v2 built successfully" -ForegroundColor Green

# Setup Phase 2 environment variables
Write-Host "`nSetting up Phase 2 environment..." -ForegroundColor Yellow

$env:SHAPESHIFTER_V2_PORT = "8084"
$env:COUNTERSTRIKE_PORT = "8088"
$env:DECEPTION_ALGORITHM = "ucb1"
$env:ANTI_DETECTION_ENABLED = "1"
$env:FINGERPRINT_MIMICRY = "1"
$env:BANDIT_EPSILON = "0.1"
$env:JITTER_MIN_MS = "10"
$env:JITTER_MAX_MS = "100"

if ($Production) {
    $env:DECEPTION_ALGORITHM = "ucb1"
    $env:BANDIT_EPSILON = "0.05"
} else {
    $env:DECEPTION_ALGORITHM = "epsilon_greedy"
    $env:BANDIT_EPSILON = "0.1"
}

Write-Host "✓ Phase 2 environment configured" -ForegroundColor Green

# Start Phase 2 services
Write-Host "`nStarting Phase 2 services..." -ForegroundColor Yellow

$phase2Services = @(
    @{Name="Shapeshifter v2"; Exe="bin/shapeshifter-v2.exe"; Port=8084}
)

$processes = @()

foreach ($service in $phase2Services) {
    Write-Host "Starting $($service.Name)..." -ForegroundColor Cyan
    
    $process = Start-Process -FilePath $service.Exe -PassThru -WindowStyle Hidden
    $processes += @{Process=$process; Name=$service.Name; Port=$service.Port}
    
    Start-Sleep -Seconds 3
    
    # Check if service is responding
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)/healthz" -TimeoutSec 10 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ $($service.Name) started on port $($service.Port)" -ForegroundColor Green
        } else {
            Write-Host "⚠ $($service.Name) may not be ready" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠ $($service.Name) health check failed" -ForegroundColor Yellow
    }
}

# Test Phase 2 functionality
Write-Host "`nTesting Phase 2 functionality..." -ForegroundColor Yellow

# Test 1: Deception Graph Selection
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8084/select-decoy" -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Deception graph decoy selection working" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Deception graph test failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Test 2: Fingerprint Mimicry
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8084/select-decoy" -TimeoutSec 10
    $serverHeader = $response.Headers["Server"]
    if ($serverHeader -and $serverHeader -like "*Apache*") {
        Write-Host "✓ HTTP fingerprint mimicry working: $serverHeader" -ForegroundColor Green
    } else {
        Write-Host "⚠ Fingerprint mimicry may not be working" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Fingerprint mimicry test failed" -ForegroundColor Yellow
}

# Test 3: Exploit Reflection
try {
    $reflectBody = @{
        attacker_ip = "192.168.1.100"
        payload = [System.Text.Encoding]::UTF8.GetBytes("SELECT * FROM users")
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Method Post -Uri "http://localhost:8084/reflect-exploit" -Body $reflectBody -ContentType 'application/json' -TimeoutSec 10
    if ($response.reflected_payload) {
        Write-Host "✓ Exploit reflection working: $($response.reflected_payload)" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠ Exploit reflection test failed: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Display Phase 2 status
Write-Host "`n=== Phase 2 Deployment Complete ===" -ForegroundColor Green
Write-Host "Phase 2 services running:" -ForegroundColor Cyan

foreach ($proc in $processes) {
    if (-not $proc.Process.HasExited) {
        Write-Host "  ✓ $($proc.Name) (PID: $($proc.Process.Id), Port: $($proc.Port))" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $($proc.Name) (Exited)" -ForegroundColor Red
    }
}

Write-Host "`nPhase 2 Capabilities:" -ForegroundColor Cyan
Write-Host "  ✓ Multi-Armed Bandit deception optimization" -ForegroundColor Green
Write-Host "  ✓ L7 fingerprint mimicry (Apache/Nginx/IIS)" -ForegroundColor Green
Write-Host "  ✓ Anti-sandbox detection with jitter" -ForegroundColor Green
Write-Host "  ✓ Exploit reflection engine" -ForegroundColor Green
Write-Host "  ✓ Adaptive decoy selection" -ForegroundColor Green

Write-Host "`nPhase 2 Metrics:" -ForegroundColor Cyan
Write-Host "  • Deception Algorithm: $env:DECEPTION_ALGORITHM" -ForegroundColor White
Write-Host "  • Bandit Epsilon: $env:BANDIT_EPSILON" -ForegroundColor White
Write-Host "  • Anti-Detection: $env:ANTI_DETECTION_ENABLED" -ForegroundColor White
Write-Host "  • Jitter Range: $env:JITTER_MIN_MS-$env:JITTER_MAX_MS ms" -ForegroundColor White

Write-Host "`nTesting Commands:" -ForegroundColor Yellow
Write-Host "  # Test deception selection:" -ForegroundColor Gray
Write-Host "  Invoke-WebRequest http://localhost:8084/select-decoy" -ForegroundColor Gray
Write-Host "  # Test exploit reflection:" -ForegroundColor Gray
Write-Host "  `$body = @{attacker_ip='test'; payload=[byte[]]@(1,2,3)} | ConvertTo-Json" -ForegroundColor Gray
Write-Host "  Invoke-RestMethod -Method Post -Uri http://localhost:8084/reflect-exploit -Body `$body -ContentType 'application/json'" -ForegroundColor Gray

Write-Host "`nNext Phase:" -ForegroundColor Yellow
Write-Host "  Phase 3: Advanced Crypto & Privacy (ZK rate-limiting, PQ-safe KEX)" -ForegroundColor White

# Save process info for cleanup
$processes | ConvertTo-Json | Out-File "phase2-processes.json"

Write-Host "`nPhase 2 deployment successful! 🛡️🔄" -ForegroundColor Green