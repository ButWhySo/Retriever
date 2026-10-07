$ErrorActionPreference = "Stop"
$dataRoot = Join-Path $env:LOCALAPPDATA "StudyRetriever"
$runtimeRoot = Join-Path $dataRoot "runtime"
$python = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
$updateFlag = Join-Path $dataRoot "update-in-progress"
for ($i = 0; $i -lt 120 -and (Test-Path -LiteralPath $updateFlag); $i++) { Start-Sleep -Seconds 1 }
if (Test-Path -LiteralPath $updateFlag) { exit 75 }
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { exit 127 }
$pluginRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:STUDY_RETRIEVER_HOME = $dataRoot
$tesseract = "C:\Program Files\Tesseract-OCR"
if (Test-Path -LiteralPath $tesseract) { $env:PATH = "$tesseract;$env:PATH" }
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"
& $python -m study_retriever.cli watch
exit $LASTEXITCODE
