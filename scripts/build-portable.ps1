# PyInstaller portable build for AI-TextAnalyze
# Models are NOT bundled — place model/ next to the exe.

param(
    [string]$Python = "python"
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

Write-Host ""
Write-Host "Build complete: dist/AI-TextAnalyze/"
Write-Host "Copy model/ folder next to AI-TextAnalyze.exe before running."
