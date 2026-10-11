$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$pythonPath = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonPath)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw 'Python 3.10 or newer is required and must be available on PATH.'
    }
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the project Python environment.' }
}

& $pythonPath -c 'import fastapi, uvicorn, multipart, PIL, pydicom, numpy, cv2, torch, torchvision, ultralytics' *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing backend dependencies from backend/requirements.txt.'
    & $pythonPath -m pip install -r backend\requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
}

if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) {
    Push-Location (Join-Path $root 'frontend')
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    } finally {
        Pop-Location
    }
}

$apiProcess = $null
try {
    $apiReady = $false
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2
        $apiReady = $health.status -eq 'ready'
    } catch { }

    if ($apiReady) {
        Write-Host 'Using the ready analysis API at http://127.0.0.1:8000'
    } else {
        $apiProcess = Start-Process -FilePath $pythonPath `
            -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', '8000') `
            -WorkingDirectory $root -PassThru

        $deadline = [DateTime]::UtcNow.AddSeconds(120)
        do {
            if ($apiProcess.HasExited) {
                throw 'The API stopped during startup. Check its console for checkpoint or dependency errors.'
            }
            try {
                $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2
                $apiReady = $health.status -eq 'ready'
            } catch { }
            if (-not $apiReady) { Start-Sleep -Milliseconds 500 }
        } while (-not $apiReady -and [DateTime]::UtcNow -lt $deadline)

        if (-not $apiReady) { throw 'The API did not become ready within 120 seconds. Check its console for startup errors.' }
        Write-Host 'Analysis API and model checkpoints are ready.'
    }

    Push-Location (Join-Path $root 'frontend')
    try {
        npm run dev -- --host 127.0.0.1
        if ($LASTEXITCODE -ne 0) { throw 'The frontend development server stopped with an error.' }
    } finally {
        Pop-Location
    }
} finally {
    if ($null -ne $apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id -Force
    }
}