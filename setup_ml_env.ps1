# =============================================================================
# PlantCare AI — Automated Python 3.12 ML Environment Setup Script
# Phase 2 Machine Learning Environment Setup
# =============================================================================

Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  PlantCare AI — Setting Up Python 3.12 Machine Learning Environment" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ""

# Check for Python 3.12 via Python Launcher (py -3.12)
$py312 = py -3.12 --version 2>$null

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($py312)) {
    Write-Host "[NOTICE] Python 3.12 was not detected via the 'py -3.12' launcher." -ForegroundColor Yellow
    Write-Host "TensorFlow on Windows officially supports Python 3.9 through 3.12." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Python 3.12 on your Windows system, choose either option:" -ForegroundColor Cyan
    Write-Host "  Option A (Recommended via Windows Terminal):" -ForegroundColor Cyan
    Write-Host "    winget install Python.Python.3.12" -ForegroundColor White
    Write-Host "  Option B (Direct Download):" -ForegroundColor Cyan
    Write-Host "    Visit https://www.python.org/downloads/release/python-3128/ and download the 64-bit installer." -ForegroundColor White
    Write-Host ""
    Write-Host "Once Python 3.12 is installed, re-run this script to create 'venv-ml'." -ForegroundColor Yellow
    Exit 1
}

Write-Host "[FOUND] Detected $py312" -ForegroundColor Green
Write-Host "[STEP 1/3] Creating dedicated virtual environment 'venv-ml'..." -ForegroundColor Cyan

# Create venv-ml without affecting the existing Flask 'venv'
py -3.12 -m venv venv-ml

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create virtual environment 'venv-ml'." -ForegroundColor Red
    Exit 1
}

Write-Host "[STEP 2/3] Upgrading pip in venv-ml..." -ForegroundColor Cyan
.\venv-ml\Scripts\python.exe -m pip install --upgrade pip

Write-Host "[STEP 3/3] Installing Phase 2 ML dependencies from requirements-ml.txt..." -ForegroundColor Cyan
.\venv-ml\Scripts\pip.exe install -r requirements-ml.txt

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  Python 3.12 ML Environment 'venv-ml' is Ready!" -ForegroundColor Green
Write-Host "  To activate in PowerShell:" -ForegroundColor White
Write-Host "    .\venv-ml\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Green
