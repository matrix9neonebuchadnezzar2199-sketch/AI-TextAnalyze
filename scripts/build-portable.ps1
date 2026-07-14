# PyInstaller onedir portable build for AI-TextAnalyze
# Fast cold start (no onefile unpack). Top-level layout:
#   dist/AI-TextAnalyze/AI-TextAnalyze.exe
#   dist/AI-TextAnalyze/model/
#   dist/AI-TextAnalyze/runtime/   (DLLs / Python / frontend — do not edit)

param(
    [string]$Python = "python",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

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

# Preserve existing model/ if re-building into the same folder
if ((Test-Path (Join-Path $DistDir "model")) -and -not (Test-Path $ModelStaging)) {
    Write-Host "Staging existing dist model/..."
    Move-Item (Join-Path $DistDir "model") $ModelStaging -Force
}

Write-Host "Building onedir AI-TextAnalyze (exe + model/ + runtime/)..."
& $Python -m PyInstaller `
    --name AI-TextAnalyze `
    --onedir `
    --contents-directory runtime `
    --windowed `
    --noconfirm `
    --clean `
    --add-data "frontend;frontend" `
    --add-data "assets;assets" `
    @IconArgs `
    --hidden-import=webview `
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

if (-not (Test-Path (Join-Path $DistDir "AI-TextAnalyze.exe"))) {
    throw "Expected onedir exe not found: $DistDir\AI-TextAnalyze.exe"
}

$ModelSrc = Join-Path $Root "model"
$ModelDst = Join-Path $DistDir "model"

if (-not $SkipModels) {
    if (Test-Path $ModelStaging) {
        Write-Host "Restoring staged model/..."
        Move-Item $ModelStaging $ModelDst -Force
    }
    elseif (Test-Path $ModelSrc) {
        Write-Host "Copying model/ next to exe (this may take a while)..."
        & robocopy $ModelSrc $ModelDst /E /NFL /NDL /NJH /NJS /nc /ns /np
        if ($LASTEXITCODE -ge 8) {
            throw "robocopy failed with exit code $LASTEXITCODE"
        }
        $global:LASTEXITCODE = 0
        Write-Host "model/ copied to $ModelDst"
    }
    else {
        Write-Warning "model/ not found — place models next to the exe before running."
    }
}
elseif (Test-Path $ModelStaging) {
    Move-Item $ModelStaging $ModelDst -Force
}

Write-Host ""
Write-Host "Build complete: dist/AI-TextAnalyze/"
Write-Host "Top-level: AI-TextAnalyze.exe + model/ + runtime/"
Get-ChildItem $DistDir | ForEach-Object {
    $kind = if ($_.PSIsContainer) { "[dir]" } else { "[file]" }
    Write-Host ("  {0} {1}" -f $kind, $_.Name)
}
