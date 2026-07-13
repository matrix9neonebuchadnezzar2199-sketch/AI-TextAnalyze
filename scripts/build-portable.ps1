# PyInstaller portable build for AI-TextAnalyze
# After build, copies model/ next to the exe (required at runtime).

param(
    [string]$Python = "python",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Installing PyInstaller if needed..."
& $Python -m pip install pyinstaller --quiet

Write-Host "Building AI-TextAnalyze.exe..."
& $Python -m PyInstaller `
    --name AI-TextAnalyze `
    --windowed `
    --noconfirm `
    --clean `
    --add-data "frontend;frontend" `
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

$DistDir = Join-Path $Root "dist\AI-TextAnalyze"
$ModelSrc = Join-Path $Root "model"
$ModelDst = Join-Path $DistDir "model"

if (-not $SkipModels) {
    if (-not (Test-Path $ModelSrc)) {
        Write-Warning "model/ not found at $ModelSrc — skip copy. Place models next to the exe before running."
    }
    else {
        Write-Host "Copying model/ next to exe (this may take a while)..."
        if (Test-Path $ModelDst) {
            Remove-Item $ModelDst -Recurse -Force
        }
        # /E サブディレクトリ込み /XD 不要なキャッシュ除外 /NFL /NDL ログ抑制 /NJH /NJS ヘッダ抑制
        & robocopy $ModelSrc $ModelDst /E /NFL /NDL /NJH /NJS /nc /ns /np
        # robocopy: 0-7 = success with optional extras
        if ($LASTEXITCODE -ge 8) {
            throw "robocopy failed with exit code $LASTEXITCODE"
        }
        $global:LASTEXITCODE = 0
        Write-Host "model/ copied to $ModelDst"
    }
}

Write-Host ""
Write-Host "Build complete: dist/AI-TextAnalyze/"
Write-Host "Layout: AI-TextAnalyze.exe + _internal/ + model/"
