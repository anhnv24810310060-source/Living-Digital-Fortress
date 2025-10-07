# Phase 5: Decentralized Verification Deployment Script

param(
    [string]$Mode = "dev"
)

Write-Host "=== ShieldX Phase 5: Decentralized Verification ===" -ForegroundColor Cyan

# Environment variables for Phase 5
$env:VERIFIER_POOL_PORT = "8087"
$env:MARKETPLACE_PORT = "8088" 
$env:MIN_VERIFIER_NODES = "3"
$env:CONSENSUS_THRESHOLD = "0.67"
$env:AUTHOR_REVENUE_PCT = "0.7"
$env:PLATFORM_REVENUE_PCT = "0.3"

# Build Phase 5 services
Write-Host "Building Phase 5 services..." -ForegroundColor Yellow

go build -o bin/verifier-pool.exe ./services/verifier-pool
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build verifier-pool" -ForegroundColor Red
    exit 1
}

go build -o bin/marketplace.exe ./services/marketplace  
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build marketplace" -ForegroundColor Red
    exit 1
}

Write-Host "Phase 5 services built successfully" -ForegroundColor Green

# Start services in background
Write-Host "Starting Phase 5 services..." -ForegroundColor Yellow

Start-Process -FilePath ".\bin\verifier-pool.exe" -WindowStyle Minimized
Start-Sleep -Seconds 2

Start-Process -FilePath ".\bin\marketplace.exe" -WindowStyle Minimized  
Start-Sleep -Seconds 2

Write-Host "Phase 5 services started:" -ForegroundColor Green
Write-Host "- Verifier Pool: http://localhost:8087" -ForegroundColor White
Write-Host "- Marketplace: http://localhost:8088" -ForegroundColor White

# Health checks
Write-Host "Performing health checks..." -ForegroundColor Yellow

$services = @(
    @{Name="Verifier Pool"; URL="http://localhost:8087/health"},
    @{Name="Marketplace"; URL="http://localhost:8088/health"}
)

foreach ($service in $services) {
    try {
        $response = Invoke-RestMethod -Uri $service.URL -TimeoutSec 5
        Write-Host "✓ $($service.Name): $($response.status)" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ $($service.Name): Failed" -ForegroundColor Red
    }
}

Write-Host "`n=== Phase 5 Deployment Complete ===" -ForegroundColor Cyan
Write-Host "Decentralized verification and community ecosystem ready!" -ForegroundColor Green