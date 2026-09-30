# CASH-AI: Python-Umgebung erstellen und Requirements installieren
# Diese Datei muss im selben Ordner wie requirements.txt liegen.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    $ProjectFolder = Split-Path -Parent $MyInvocation.MyCommand.Path
    Set-Location $ProjectFolder

    $RequirementsFile = Join-Path $ProjectFolder "requirements.txt"
    $VenvFolder = Join-Path $ProjectFolder ".venv"
    $VenvPython = Join-Path $VenvFolder "Scripts\python.exe"

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "CASH-AI: Python-Requirements installieren" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Projektordner: $ProjectFolder"

    if (-not (Test-Path $RequirementsFile -PathType Leaf)) {
        throw "Die Datei 'requirements.txt' wurde nicht gefunden. Lege dieses Skript in denselben Ordner wie requirements.txt."
    }

    # Bevorzugt den Windows-Python-Launcher 'py'. Falls dieser nicht vorhanden
    # ist, wird der Befehl 'python' verwendet.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $PythonCommand = "py"
        $PythonArguments = @("-3")
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonCommand = "python"
        $PythonArguments = @()
    }
    else {
        throw "Python wurde nicht gefunden. Installiere Python 3.11 oder neuer und aktiviere bei der Installation 'Add python.exe to PATH'."
    }

    $PythonVersionText = (& $PythonCommand @PythonArguments -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Die installierte Python-Version konnte nicht ermittelt werden."
    }

    $PythonVersion = [Version]$PythonVersionText
    if ($PythonVersion -lt [Version]"3.11") {
        throw "Gefunden wurde Python $PythonVersionText. Fuer CASH-AI wird Python 3.11 oder neuer benoetigt."
    }

    Write-Host "Python $PythonVersionText wurde gefunden." -ForegroundColor Green

    if (-not (Test-Path $VenvPython -PathType Leaf)) {
        Write-Host "Erstelle die virtuelle Umgebung .venv ..." -ForegroundColor Yellow
        & $PythonCommand @PythonArguments -m venv $VenvFolder
        if ($LASTEXITCODE -ne 0) {
            throw "Die virtuelle Python-Umgebung konnte nicht erstellt werden."
        }
    }
    else {
        Write-Host "Die vorhandene virtuelle Umgebung .venv wird verwendet." -ForegroundColor DarkGray
    }

    Write-Host "Aktualisiere pip ..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip konnte nicht aktualisiert werden."
    }

    Write-Host "Installiere die Pakete aus requirements.txt ..." -ForegroundColor Yellow
    & $VenvPython -m pip install --requirement $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        throw "Mindestens ein benoetigtes Python-Paket konnte nicht installiert werden."
    }

    Write-Host "Pruefe die installierten Bibliotheken ..." -ForegroundColor Yellow
    & $VenvPython -c "import pandas, numpy, openpyxl, matplotlib, seaborn; print('Importpruefung erfolgreich.')"
    if ($LASTEXITCODE -ne 0) {
        throw "Die Installation wurde abgeschlossen, aber die Importpruefung ist fehlgeschlagen."
    }

    Write-Host ""
    Write-Host "Alle Requirements wurden erfolgreich installiert." -ForegroundColor Green
    Write-Host ""
    Write-Host "CASH-AI kannst du danach mit diesem PowerShell-Befehl starten:"
    Write-Host '& ".\.venv\Scripts\python.exe" ".\CASH_AI_Datenaufbereitung.py"' -ForegroundColor Cyan
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "FEHLER: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    exit 1
}
