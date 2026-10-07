$ErrorActionPreference = "Stop"
$runtimeRoot = Join-Path $env:LOCALAPPDATA "StudyRetriever\runtime"
$python = Join-Path $runtimeRoot ".venv\Scripts\python.exe"
$updateFlag = Join-Path $env:LOCALAPPDATA "StudyRetriever\update-in-progress"
if (Test-Path -LiteralPath $updateFlag) {
    [Console]::Error.WriteLine("Study Retriever is being updated. Retry after the update finishes and restart ChatGPT Desktop.")
    exit 75
}
if (-not (Test-Path $python)) {
    [Console]::Error.WriteLine("Study Retriever runtime is not installed. Run scripts\\install.ps1 once from the plugin folder.")
    exit 127
}
$pluginRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:STUDY_RETRIEVER_HOME = Join-Path $env:LOCALAPPDATA "StudyRetriever"
$tesseract = "C:\Program Files\Tesseract-OCR"
if (Test-Path -LiteralPath $tesseract) { $env:PATH = "$tesseract;$env:PATH" }
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"
& $python -m study_retriever.mcp_server
exit $LASTEXITCODE
