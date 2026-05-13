param(
    [ValidateSet("menu", "setup", "backend", "frontend", "start", "train-lite", "train-balanced", "train-full")]
    [string]$Mode = "start",
    [switch]$NoRebuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$BackendVenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
$script:ResolvedBackendPython = $null

function Write-Section {
    param([string]$Text)
    Write-Host ""
    Write-Host "==== $Text ====" -ForegroundColor Cyan
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found in PATH."
    }
}

function Get-BackendPython {
    if ($null -ne $script:ResolvedBackendPython -and $script:ResolvedBackendPython -ne "") {
        return $script:ResolvedBackendPython
    }
    if (Test-Path $BackendVenvPython) {
        return $BackendVenvPython
    }
    return "python"
}

function Ensure-BackendVenv {
    Require-Command "python"
    if (-not (Test-Path $BackendVenvPython)) {
        Write-Section "Creating backend virtual environment"
        Push-Location $BackendDir
        try {
            & python -m venv .venv
        } finally {
            Pop-Location
        }
    }
}

function Install-BackendDeps {
    Write-Section "Installing backend dependencies"
    Ensure-BackendVenv
    $py = Get-BackendPython
    Push-Location $BackendDir
    try {
        & $py -m pip install --upgrade pip
        & $py -m pip install -r requirements.txt
    } finally {
        Pop-Location
    }
}

function Test-BackendDepsInstalled {
    param([string]$PythonPath = "")
    if ($PythonPath -eq "") {
        $PythonPath = Get-BackendPython
    }
    Push-Location $BackendDir
    try {
        try {
            & $PythonPath -c "import numpy, scipy, fastapi, torch" *> $null
            return ($LASTEXITCODE -eq 0)
        } catch {
            return $false
        }
    } finally {
        Pop-Location
    }
}

function Ensure-BackendReady {
    Require-Command "python"
    $candidates = @()
    if (Test-Path $BackendVenvPython) {
        $candidates += $BackendVenvPython
    }
    $candidates += "python"

    foreach ($candidate in $candidates) {
        if (Test-BackendDepsInstalled -PythonPath $candidate) {
            $script:ResolvedBackendPython = $candidate
            return
        }
    }

    $installTarget = if (Test-Path $BackendVenvPython) { $BackendVenvPython } else { "python" }
    if (-not (Test-BackendDepsInstalled -PythonPath $installTarget)) {
        Write-Section "Backend dependencies missing; installing requirements"
        Push-Location $BackendDir
        try {
            & $installTarget -m pip install -r requirements.txt
        } finally {
            Pop-Location
        }
    }

    if (-not (Test-BackendDepsInstalled -PythonPath $installTarget)) {
        throw "Backend dependencies are still missing after install. Run '.\\startup.ps1 -Mode setup' and retry."
    }
    $script:ResolvedBackendPython = $installTarget
}

function Install-FrontendDeps {
    Write-Section "Installing frontend dependencies"
    Require-Command "npm"
    Push-Location $FrontendDir
    try {
        & npm install
    } finally {
        Pop-Location
    }
}

function Setup-All {
    Install-BackendDeps
    if (Test-BackendDepsInstalled -PythonPath $BackendVenvPython) {
        $script:ResolvedBackendPython = $BackendVenvPython
    }
    Install-FrontendDeps
    Write-Host "Setup complete." -ForegroundColor Green
}

function Escape-SingleQuoted {
    param([string]$Value)
    return $Value -replace "'", "''"
}

function Start-Backend {
    Ensure-BackendReady
    $py = Get-BackendPython
    $safeBackend = Escape-SingleQuoted -Value $BackendDir
    $safePy = Escape-SingleQuoted -Value $py
    $cmd = "Set-Location '$safeBackend'; & '$safePy' -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

    Write-Section "Starting backend in a new PowerShell window"
    Start-Process -FilePath "powershell" -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd) | Out-Null
}

function Start-Frontend {
    Require-Command "npm"
    if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
        Write-Section "Frontend node_modules not found. Installing dependencies..."
        Push-Location $FrontendDir
        try {
            & npm install
        } finally {
            Pop-Location
        }
    }
    $safeFrontend = Escape-SingleQuoted -Value $FrontendDir
    $cmd = "Set-Location '$safeFrontend'; `$env:VITE_API_BASE_URL='http://localhost:8000'; npm run dev"

    Write-Section "Starting frontend in a new PowerShell window"
    Start-Process -FilePath "powershell" -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $cmd) | Out-Null
}

function Start-Stack {
    Ensure-BackendVenv
    Start-Backend
    Start-Frontend
    Write-Host "Backend and frontend windows launched." -ForegroundColor Green
}

function Set-LaptopTrainingEnv {
    # Keep CPU usage modest to avoid making the laptop unresponsive.
    $env:OMP_NUM_THREADS = "2"
    $env:MKL_NUM_THREADS = "2"
    $env:OPENBLAS_NUM_THREADS = "2"
    $env:NUMEXPR_NUM_THREADS = "2"
    $env:VECLIB_MAXIMUM_THREADS = "2"
    $env:TOKENIZERS_PARALLELISM = "false"
}

function Invoke-BackendPythonBelowNormal {
    param([string[]]$Arguments)
    Ensure-BackendReady
    $py = Get-BackendPython

    Push-Location $BackendDir
    try {
        $proc = Start-Process -FilePath $py -ArgumentList $Arguments -PassThru -NoNewWindow
        Start-Sleep -Milliseconds 250
        if (-not $proc.HasExited) {
            try {
                $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
            } catch {
                # Priority change can fail on some systems; continue anyway.
            }
        }
        $proc.WaitForExit()
        $proc.Refresh()
        if ($null -eq $proc.ExitCode -or $proc.ExitCode -ne 0) {
            throw "Training failed with exit code $($proc.ExitCode)."
        }
    } finally {
        Pop-Location
    }
}

function Build-TrainArgs {
    param(
        [string]$Profile
    )
    $args = @("-m", "app.training.train_all")
    if (-not $NoRebuild) {
        $args += "--rebuild-dataset"
    }

    switch ($Profile) {
        "lite" {
            # Lowest-impact profile for local laptops.
            $args += @("--tiny-mode", "--no-ptbxl", "--epochs", "1", "--batch-size", "24", "--skip-classical")
        }
        "balanced" {
            # Still laptop-friendly, trains both deep and classical quickly.
            $args += @("--quick-mode", "--no-ptbxl", "--epochs", "2", "--batch-size", "64")
        }
        "full" {
            # Heavy profile. Can significantly load CPU/GPU.
            $args += @("--epochs", "30", "--batch-size", "256")
        }
        default {
            throw "Unknown training profile: $Profile"
        }
    }
    return $args
}

function Run-Training {
    param([string]$Profile)
    Set-LaptopTrainingEnv
    $trainArgs = Build-TrainArgs -Profile $Profile
    Write-Section "Running training profile: $Profile"
    Write-Host "Command: python $($trainArgs -join ' ')" -ForegroundColor DarkGray
    Invoke-BackendPythonBelowNormal -Arguments $trainArgs
    Write-Host "Training finished." -ForegroundColor Green
}

function Show-Menu {
    while ($true) {
        Write-Host ""
        Write-Host "ECG Arrhythmia Project Launcher" -ForegroundColor Yellow
        Write-Host "1. Setup dependencies (backend + frontend)"
        Write-Host "2. Start backend API"
        Write-Host "3. Start frontend dashboard"
        Write-Host "4. Start backend + frontend"
        Write-Host "5. Train (Laptop Lite - recommended)"
        Write-Host "6. Train (Balanced Quick)"
        Write-Host "7. Train (Full Heavy)"
        Write-Host "8. Exit"
        $choice = Read-Host "Choose an option"

        switch ($choice) {
            "1" { Setup-All }
            "2" { Start-Backend }
            "3" { Start-Frontend }
            "4" { Start-Stack }
            "5" { Run-Training -Profile "lite" }
            "6" { Run-Training -Profile "balanced" }
            "7" { Run-Training -Profile "full" }
            "8" { break }
            default { Write-Host "Invalid choice. Try again." -ForegroundColor Red }
        }
    }
}

switch ($Mode) {
    "menu" { Show-Menu }
    "setup" { Setup-All }
    "backend" { Start-Backend }
    "frontend" { Start-Frontend }
    "start" { Start-Stack }
    "train-lite" { Run-Training -Profile "lite" }
    "train-balanced" { Run-Training -Profile "balanced" }
    "train-full" { Run-Training -Profile "full" }
}
