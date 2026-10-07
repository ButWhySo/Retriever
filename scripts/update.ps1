param(
    [string]$ReleaseZip = "",
    [string]$ManifestUrl = "",
    [string]$ExpectedSha256 = "",
    [switch]$SaveFeed,
    [switch]$CheckOnly,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Write-Step([string]$Message) {
    Write-Host "[Study Retriever Update] $Message"
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Test-Sha256String([string]$Value) {
    return $Value -match '^[0-9A-Fa-f]{64}$'
}

function Read-ReleaseMetadata([string]$Root) {
    $rootFull = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    $path = Join-Path $rootFull "release.json"
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Release metadata missing: $path"
    }
    try {
        $meta = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
    } catch {
        throw "Invalid release.json: $($_.Exception.Message)"
    }
    if ($meta.name -ne "study-retriever") { throw "release.json name must be study-retriever" }
    if (-not ($meta.version -as [version])) { throw "release.json version is not valid stable semver: $($meta.version)" }
    if (-not $meta.wheel -or -not $meta.wheel.file -or -not $meta.wheel.sha256) {
        throw "release.json must declare wheel.file and wheel.sha256"
    }
    if (-not (Test-Sha256String ([string]$meta.wheel.sha256))) {
        throw "release.json wheel.sha256 is invalid"
    }
    $wheelPath = [IO.Path]::GetFullPath((Join-Path $rootFull ([string]$meta.wheel.file)))
    if (-not ($wheelPath.StartsWith($rootFull + '\', [StringComparison]::OrdinalIgnoreCase))) {
        throw "release.json wheel path escapes the release directory"
    }
    if (-not (Test-Path -LiteralPath $wheelPath -PathType Leaf)) {
        throw "Bundled release wheel is missing: $wheelPath"
    }
    $actual = Get-Sha256 $wheelPath
    if ($actual -ne ([string]$meta.wheel.sha256).ToLowerInvariant()) {
        throw "Bundled Study Retriever wheel SHA-256 mismatch"
    }
    return $meta
}

function Get-InstalledVersion([string]$PluginRoot) {
    $releasePath = Join-Path $PluginRoot "release.json"
    if (Test-Path -LiteralPath $releasePath -PathType Leaf) {
        try {
            $release = Get-Content -Raw -LiteralPath $releasePath | ConvertFrom-Json
            if ($release.version -as [version]) { return [version]$release.version }
        } catch { }
    }
    $pluginPath = Join-Path $PluginRoot "plugin.json"
    if (Test-Path -LiteralPath $pluginPath -PathType Leaf) {
        try {
            $plugin = Get-Content -Raw -LiteralPath $pluginPath | ConvertFrom-Json
            if ($plugin.version -as [version]) { return [version]$plugin.version }
        } catch { }
    }
    return [version]"0.0.0"
}

function Test-SafeZip([string]$ZipPath) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($entry in $archive.Entries) {
            $name = $entry.FullName.Replace('/', '\')
            if ([IO.Path]::IsPathRooted($name)) { throw "Unsafe absolute path in update ZIP: $name" }
            $parts = $name.Split('\')
            if ($parts -contains '..') { throw "Unsafe parent traversal in update ZIP: $name" }
        }
    } finally {
        $archive.Dispose()
    }
}

function Find-ReleaseRoot([string]$ExtractRoot) {
    if (Test-Path -LiteralPath (Join-Path $ExtractRoot "release.json") -PathType Leaf) {
        return [IO.Path]::GetFullPath($ExtractRoot)
    }
    $matches = @(Get-ChildItem -LiteralPath $ExtractRoot -Filter release.json -File -Recurse -ErrorAction Stop)
    if ($matches.Count -ne 1) {
        throw "Update ZIP must contain exactly one release.json; found $($matches.Count)"
    }
    return $matches[0].Directory.FullName
}

function Mirror-Directory([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $null = & robocopy.exe $Source $Destination /MIR /R:2 /W:1 /NFL /NDL /NJH /NJS /NP /XD .git .venv .study-data __pycache__ .pytest_cache build dist /XF *.pyc
    if ($LASTEXITCODE -gt 7) { throw "robocopy failed with exit code $LASTEXITCODE" }
}

function Stop-StudyRetrieverProcesses([string]$RuntimeRoot) {
    $runtimeFull = [IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\') + '\'
    try {
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
            $_.ProcessId -ne $PID -and (
                ($_.ExecutablePath -and ([IO.Path]::GetFullPath($_.ExecutablePath).StartsWith($runtimeFull, [StringComparison]::OrdinalIgnoreCase))) -or
                ($_.CommandLine -and $_.CommandLine -match 'study_retriever\.(mcp_server|cli)')
            )
        })
        foreach ($proc in $processes) {
            Write-Step "Stopping Study Retriever process PID $($proc.ProcessId) for atomic runtime update"
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        }
    } catch {
        Write-Step "Could not enumerate/stop all existing Study Retriever processes: $($_.Exception.Message)"
    }
}

function Read-RemoteManifest([string]$Url) {
    $uri = [Uri]$Url
    if ($uri.Scheme -ne 'https') { throw "Update manifest URL must use HTTPS" }
    $response = Invoke-WebRequest -UseBasicParsing -Uri $uri.AbsoluteUri -Method Get
    $manifest = $response.Content | ConvertFrom-Json
    if ($manifest.name -ne 'study-retriever') { throw "Update manifest name must be study-retriever" }
    if (-not ($manifest.version -as [version])) { throw "Update manifest version is invalid" }
    if (-not $manifest.url) { throw "Update manifest is missing url" }
    if (-not $manifest.sha256 -or -not (Test-Sha256String ([string]$manifest.sha256))) {
        throw "Update manifest is missing a valid SHA-256"
    }
    $downloadUri = [Uri]::new($uri, [string]$manifest.url)
    if ($downloadUri.Scheme -ne 'https') { throw "Update package URL must use HTTPS" }
    return [pscustomobject]@{
        version = [string]$manifest.version
        url = $downloadUri.AbsoluteUri
        sha256 = ([string]$manifest.sha256).ToLowerInvariant()
    }
}

$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dataRoot = Join-Path $env:LOCALAPPDATA "StudyRetriever"
$runtimeRoot = Join-Path $dataRoot "runtime"
$pluginRoot = Join-Path $HOME ".codex\plugins\study-retriever"
$feedConfig = Join-Path $dataRoot "update-feed.json"
$updateFlag = Join-Path $dataRoot "update-in-progress"
$lockPath = Join-Path $dataRoot "update.lock"

New-Item -ItemType Directory -Force -Path $dataRoot | Out-Null

if ($SaveFeed -and -not $ManifestUrl) {
    throw "-SaveFeed requires -ManifestUrl"
}
if ($ReleaseZip -and $ManifestUrl) {
    throw "Use either -ReleaseZip or -ManifestUrl, not both"
}
if ($ExpectedSha256 -and -not $ReleaseZip) {
    throw "-ExpectedSha256 is only valid with -ReleaseZip"
}

$effectiveManifestUrl = $ManifestUrl
if (-not $ReleaseZip -and -not $effectiveManifestUrl -and (Test-Path -LiteralPath $feedConfig -PathType Leaf)) {
    try {
        $saved = Get-Content -Raw -LiteralPath $feedConfig | ConvertFrom-Json
        $effectiveManifestUrl = [string]$saved.manifest_url
    } catch {
        throw "Saved update feed is invalid: $feedConfig"
    }
}

$lockStream = $null
$tempRoot = $null
$backupRoot = $null
$rollbackAvailable = $false
try {
    try {
        $lockStream = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    } catch {
        throw "Another Study Retriever update is already running"
    }

    $candidateRoot = $sourceRoot
    $remoteManifest = $null
    $currentVersion = Get-InstalledVersion $pluginRoot

    if ($effectiveManifestUrl) {
        Write-Step "Checking update feed $effectiveManifestUrl"
        $remoteManifest = Read-RemoteManifest $effectiveManifestUrl
        $remoteVersion = [version]$remoteManifest.version
        if ($SaveFeed) {
            [IO.File]::WriteAllText(
                $feedConfig,
                (([pscustomobject]@{ manifest_url = $ManifestUrl } | ConvertTo-Json) + [Environment]::NewLine),
                [Text.UTF8Encoding]::new($false)
            )
        }
        Write-Step "Installed version: $currentVersion"
        Write-Step "Feed version: $remoteVersion"
        if ($CheckOnly) {
            if ($remoteVersion -gt $currentVersion) { Write-Host "Update available: $currentVersion -> $remoteVersion" }
            else { Write-Host "No newer Study Retriever release is available." }
            return
        }
        if (-not $Force -and $remoteVersion -le $currentVersion) {
            Write-Host "Study Retriever $currentVersion is already current."
            return
        }
    }

    if ($ReleaseZip -or $effectiveManifestUrl) {
        $tempRoot = Join-Path $env:TEMP ("study-retriever-update-" + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
        $zipPath = Join-Path $tempRoot "release.zip"

        if ($effectiveManifestUrl) {
            Write-Step "Downloading Study Retriever $($remoteManifest.version)"
            Invoke-WebRequest -UseBasicParsing -Uri $remoteManifest.url -OutFile $zipPath
            $expected = $remoteManifest.sha256
        } else {
            $resolvedZip = (Resolve-Path -LiteralPath $ReleaseZip).Path
            Copy-Item -LiteralPath $resolvedZip -Destination $zipPath
            $expected = $ExpectedSha256.ToLowerInvariant()
        }

        if ($expected) {
            if (-not (Test-Sha256String $expected)) { throw "Expected package SHA-256 is invalid" }
            $actualZip = Get-Sha256 $zipPath
            if ($actualZip -ne $expected) { throw "Update ZIP SHA-256 mismatch; refusing to install" }
        }

        Test-SafeZip $zipPath
        $extractRoot = Join-Path $tempRoot "extracted"
        New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
        Expand-Archive -LiteralPath $zipPath -DestinationPath $extractRoot -Force
        $candidateRoot = Find-ReleaseRoot $extractRoot
    }

    $release = Read-ReleaseMetadata $candidateRoot
    $newVersion = [version]$release.version

    if ($remoteManifest -and ([string]$release.version -ne $remoteManifest.version)) {
        throw "Remote manifest version $($remoteManifest.version) does not match package version $($release.version)"
    }

    Write-Step "Installed version: $currentVersion"
    Write-Step "Candidate version: $newVersion"

    if ($CheckOnly) {
        if ($newVersion -gt $currentVersion) {
            Write-Host "Update available: $currentVersion -> $newVersion"
            return
        }
        Write-Host "No newer Study Retriever release is available."
        return
    }

    if (-not $Force -and $newVersion -le $currentVersion) {
        Write-Host "Study Retriever $currentVersion is already current. Use -Force only to reinstall the same release."
        return
    }

    [IO.File]::WriteAllText($updateFlag, "updating", [Text.Encoding]::ASCII)
    Stop-StudyRetrieverProcesses $runtimeRoot

    if (Test-Path -LiteralPath $pluginRoot -PathType Container) {
        $backupRoot = Join-Path $env:TEMP ("study-retriever-rollback-" + [Guid]::NewGuid().ToString('N'))
        Write-Step "Creating rollback copy of installed plugin"
        Mirror-Directory $pluginRoot $backupRoot
        $rollbackAvailable = $true
    }

    try {
        Write-Step "Installing and live-verifying Study Retriever $newVersion"
        & (Join-Path $candidateRoot "scripts\install.ps1")
        Write-Step "Update installed successfully"
    } catch {
        $installFailure = $_
        if ($rollbackAvailable) {
            Write-Step "Update failed; restoring previous release"
            $rollbackFailure = $null
            try {
                Mirror-Directory $backupRoot $pluginRoot
                & (Join-Path $backupRoot "scripts\install.ps1") -SkipMarketplace
            } catch {
                $rollbackFailure = $_
            }
            if ($rollbackFailure) {
                throw "Update failed and automatic rollback also failed. Update error: $($installFailure.Exception.Message). Rollback error: $($rollbackFailure.Exception.Message)"
            }
            throw "Update failed and previous release was restored. Original error: $($installFailure.Exception.Message)"
        }
        throw $installFailure
    }

    Write-Host ""
    Write-Host "Study Retriever updated to $newVersion. Restart ChatGPT Desktop so its local-plugin cache reloads the new release."
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $updateFlag
    if ($lockStream) { $lockStream.Dispose() }
    if ($tempRoot -and (Test-Path -LiteralPath $tempRoot)) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $tempRoot }
    if ($backupRoot -and (Test-Path -LiteralPath $backupRoot)) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $backupRoot }
}
