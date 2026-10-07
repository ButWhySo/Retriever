param(
    [switch]$DeleteIndexAndModels
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$personalMarketplaceDir = Join-Path $HOME ".agents\plugins"
$pluginRoot = Join-Path $HOME ".codex\plugins\study-retriever"
$marketFile = Join-Path $personalMarketplaceDir "marketplace.json"
$dataRoot = Join-Path $env:LOCALAPPDATA "StudyRetriever"
$runtimeRoot = Join-Path $dataRoot "runtime"

$schtasks = Get-Command schtasks.exe -ErrorAction SilentlyContinue
if ($schtasks) {
    & $schtasks.Source /End /TN "Study Retriever Background Indexer" 2>$null | Out-Null
    & $schtasks.Source /Delete /TN "Study Retriever Background Indexer" /F 2>$null | Out-Null
}
Remove-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "StudyRetrieverIndexer" -ErrorAction SilentlyContinue

if (Test-Path $marketFile) {
    $market = Get-Content -Raw $marketFile | ConvertFrom-Json
    if ($market.PSObject.Properties.Name -contains "plugins") {
        $market.plugins = @($market.plugins | Where-Object { $_.name -ne "study-retriever" })
        [IO.File]::WriteAllText($marketFile, (($market | ConvertTo-Json -Depth 20) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    }
}

# Installed plugin code and Desktop's local-plugin caches are disposable copies.
foreach ($path in @($pluginRoot, $runtimeRoot)) {
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}
$pluginCacheRoot = Join-Path $HOME ".codex\plugins\cache"
if (Test-Path -LiteralPath $pluginCacheRoot -PathType Container) {
    foreach ($marketCache in @(Get-ChildItem -LiteralPath $pluginCacheRoot -Directory -ErrorAction SilentlyContinue)) {
        $pluginCache = Join-Path $marketCache.FullName "study-retriever"
        if (Test-Path -LiteralPath $pluginCache) { Remove-Item -Recurse -Force -LiteralPath $pluginCache }
    }
}

if ($DeleteIndexAndModels -and (Test-Path $dataRoot)) {
    Remove-Item -Recurse -Force $dataRoot
}

Write-Host "Study Retriever removed. Restart ChatGPT Desktop."
if (-not $DeleteIndexAndModels) {
    Write-Host "Index, model cache, and configured root metadata were retained under $dataRoot for a future reinstall."
}
Write-Host "Original study files were never modified or deleted."
