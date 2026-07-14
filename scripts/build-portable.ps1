# PyInstaller onefile portable build for AI-TextAnalyze
# Distribution top-level (only these two):
#   dist/AI-TextAnalyze/AI-TextAnalyze.exe
#   dist/AI-TextAnalyze/models/

param(
    [string]$Python = "",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

# 既定はプロジェクト venv（システム python には pywebview が無いことがある）
if (-not $Python) {
    $VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $Python = $VenvPython
    }
    else {
        $Python = "python"
    }
}

Write-Host "Using Python: $Python"
& $Python -c "import webview; print('webview OK:', webview.__file__)"
if ($LASTEXITCODE -ne 0) {
    throw "pywebview is not installed in this Python. Run: $Python -m pip install -r requirements.txt"
}

Write-Host "Installing PyInstaller if needed..."
& $Python -m pip install "pyinstaller>=6.3" --quiet

$IconPath = Join-Path $Root "assets\AI-TextAnalyze.ico"
$IconArgs = @()
if (Test-Path $IconPath) {
    $IconArgs = @("--icon", $IconPath)
    Write-Host "Using icon: $IconPath"
}
else {
    Write-Warning "Icon not found at $IconPath — building without custom icon."
}

$DistDir = Join-Path $Root "dist\AI-TextAnalyze"
$ModelStaging = Join-Path $Root "dist\_model_staging"

# Preserve existing models/ or legacy model/ when rebuilding
foreach ($name in @("models", "model")) {
    $existing = Join-Path $DistDir $name
    if ((Test-Path $existing) -and -not (Test-Path $ModelStaging)) {
        Write-Host "Staging existing dist $name/..."
        Move-Item $existing $ModelStaging -Force
        break
    }
}

Write-Host "Building onefile AI-TextAnalyze.exe (models/ beside exe)..."
& $Python -m PyInstaller `
    --name AI-TextAnalyze `
    --onefile `
    --windowed `
    --noconfirm `
    --clean `
    --add-data "frontend;frontend" `
    --add-data "assets;assets" `
    @IconArgs `
    --collect-all webview `
    --hidden-import=webview `
    --hidden-import=webview.platforms `
    --hidden-import=webview.platforms.edgechromium `
    --hidden-import=backend.api `
    --hidden-import=backend.model_manager `
    --hidden-import=backend.ner_engine `
    --hidden-import=backend.mt_engine `
    --hidden-import=backend.pdf_reader `
    --hidden-import=backend.lang_detect `
    app.py
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$BuiltExe = Join-Path $Root "dist\AI-TextAnalyze.exe"
if (-not (Test-Path $BuiltExe)) {
    throw "Expected onefile exe not found: $BuiltExe"
}

if (Test-Path $DistDir) {
    Remove-Item $DistDir -Recurse -Force
}
New-Item -ItemType Directory -Path $DistDir | Out-Null
Move-Item $BuiltExe (Join-Path $DistDir "AI-TextAnalyze.exe") -Force

$ModelSrc = Join-Path $Root "model"
$ModelDst = Join-Path $DistDir "models"

if (-not $SkipModels) {
    if (Test-Path $ModelStaging) {
        Write-Host "Restoring staged models/..."
        Move-Item $ModelStaging $ModelDst -Force
    }
    elseif (Test-Path $ModelSrc) {
        Write-Host "Copying model/ -> models/ next to exe (this may take a while)..."
        & robocopy $ModelSrc $ModelDst /E /NFL /NDL /NJH /NJS /nc /ns /np
        if ($LASTEXITCODE -ge 8) {
            throw "robocopy failed with exit code $LASTEXITCODE"
        }
        $global:LASTEXITCODE = 0
        Write-Host "models/ copied to $ModelDst"
    }
    else {
        Write-Warning "model/ not found — place models next to the exe before running."
    }
}
elseif (Test-Path $ModelStaging) {
    Move-Item $ModelStaging $ModelDst -Force
}

$Top = @(Get-ChildItem $DistDir | ForEach-Object { $_.Name })
$Unexpected = $Top | Where-Object { $_ -notin @("AI-TextAnalyze.exe", "models") }
if ($Unexpected) {
    Write-Warning ("Unexpected top-level entries: {0}" -f ($Unexpected -join ", "))
}

Write-Host ""
Write-Host "Build complete: dist/AI-TextAnalyze/"
Write-Host "Top-level: AI-TextAnalyze.exe + models/"
Get-ChildItem $DistDir | ForEach-Object {
    $kind = if ($_.PSIsContainer) { "[dir]" } else { "[file]" }
    Write-Host ("  {0} {1}" -f $kind, $_.Name)
}
