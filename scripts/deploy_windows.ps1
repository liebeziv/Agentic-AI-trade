# =============================================================================
# atlas-trader — Windows PowerShell production deployment script
# Usage: .\scripts\deploy_windows.ps1
# Requires: Docker Desktop for Windows, PowerShell 5.1+
# =============================================================================
#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
function Write-Info  { param([string]$msg) Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Write-Warn  { param([string]$msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }
function Exit-Fail   { param([string]$msg) Write-Err $msg; exit 1 }

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Info "Project root: $ProjectRoot"
Set-Location $ProjectRoot

# ---------------------------------------------------------------------------
# 1. Prerequisite checks
# ---------------------------------------------------------------------------
Write-Info "Checking prerequisites..."

if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Exit-Fail "docker not found. Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
}

# Prefer 'docker compose' (v2 plugin) then fall back to 'docker-compose' (v1)
$dc = $null
try {
    docker compose version | Out-Null
    $dc = "docker compose"
} catch {
    if (Get-Command "docker-compose" -ErrorAction SilentlyContinue) {
        $dc = "docker-compose"
    } else {
        Exit-Fail "docker-compose not found. Upgrade Docker Desktop or install Compose v1."
    }
}
Write-Info "Using compose command: $dc"

if (-not (Test-Path ".env")) {
    Exit-Fail ".env file not found. Copy .env.example to .env and fill in your values:`n  Copy-Item .env.example .env"
}

# ---------------------------------------------------------------------------
# 2. Required environment variable validation
# ---------------------------------------------------------------------------
Write-Info "Validating required environment variables..."

# Parse .env (simple KEY=VALUE, skip comments and blank lines)
$envVars = @{}
Get-Content ".env" | Where-Object { $_ -match "^\s*[^#]" -and $_ -match "=" } | ForEach-Object {
    $parts = $_ -split "=", 2
    if ($parts.Length -eq 2) {
        $envVars[$parts[0].Trim()] = $parts[1].Trim()
    }
}

$required = @("ANTHROPIC_API_KEY", "ALPACA_API_KEY")
$missing  = @()

foreach ($var in $required) {
    $val = $envVars[$var]
    if ([string]::IsNullOrWhiteSpace($val) -or $val -like "your_*" -or $val -like "sk-ant-...*") {
        $missing += $var
    }
}

if ($missing.Count -gt 0) {
    Write-Err "The following required variables are not set in .env:"
    $missing | ForEach-Object { Write-Err "  - $_" }
    Exit-Fail "Fix .env before deploying."
}

Write-Info "All required environment variables are present."

# ---------------------------------------------------------------------------
# 3. Pull latest code
# ---------------------------------------------------------------------------
Write-Info "Pulling latest code from remote..."
try {
    git pull --ff-only
    Write-Info "Code updated successfully."
} catch {
    Write-Warn "git pull failed (maybe no remote, or local changes). Continuing with current code."
}

# ---------------------------------------------------------------------------
# 4. Build Docker images (no cache)
# ---------------------------------------------------------------------------
Write-Info "Building Docker images (--no-cache)..."
Invoke-Expression "$dc build --no-cache"
if ($LASTEXITCODE -ne 0) { Exit-Fail "Docker build failed." }
Write-Info "Build complete."

# ---------------------------------------------------------------------------
# 5. Stop existing containers gracefully
# ---------------------------------------------------------------------------
Write-Info "Stopping existing containers (if any)..."
Invoke-Expression "$dc down --remove-orphans" 2>&1 | Out-Null
Write-Info "Existing containers stopped."

# ---------------------------------------------------------------------------
# 6. Start in production mode (detached)
# ---------------------------------------------------------------------------
Write-Info "Starting atlas-trader in production mode..."
Invoke-Expression "$dc up -d"
if ($LASTEXITCODE -ne 0) { Exit-Fail "Failed to start containers." }
Write-Info "Containers started."

# ---------------------------------------------------------------------------
# 7. Health check — wait up to 30 s
# ---------------------------------------------------------------------------
$healthUrl = "http://localhost:8080/health"
$maxWait   = 30
$interval  = 3
$elapsed   = 0
$healthy   = $false

Write-Info "Waiting for health check at $healthUrl (timeout: ${maxWait}s)..."

while ($elapsed -lt $maxWait) {
    try {
        $resp = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) {
            $healthy = $true
            break
        }
    } catch {
        # not ready yet — keep waiting
    }
    Start-Sleep -Seconds $interval
    $elapsed += $interval
    Write-Info "  Still waiting... (${elapsed}s elapsed)"
}

# ---------------------------------------------------------------------------
# 8. Show container status and recent logs
# ---------------------------------------------------------------------------
Write-Info "Container status:"
Invoke-Expression "$dc ps"

Write-Info "Recent container logs:"
Write-Host ("=" * 70)
Invoke-Expression "$dc logs --tail=20"
Write-Host ("=" * 70)

# ---------------------------------------------------------------------------
# 9. Final status
# ---------------------------------------------------------------------------
if ($healthy) {
    Write-Info "Health check PASSED after ~${elapsed}s."
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Green
    Write-Host "  atlas-trader deployed successfully!" -ForegroundColor Green
    Write-Host "  Trader API : http://localhost:8080"  -ForegroundColor Green
    Write-Host "  Dashboard  : http://localhost:8501"  -ForegroundColor Green
    Write-Host ("=" * 60) -ForegroundColor Green
    exit 0
} else {
    Write-Err "Health check FAILED after ${maxWait}s."
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Red
    Write-Host "  Deployment may have issues. Check the logs above." -ForegroundColor Red
    Write-Host "  Run: $dc logs -f trader"                           -ForegroundColor Red
    Write-Host ("=" * 60) -ForegroundColor Red
    exit 1
}
