param(
    [Parameter(ValueFromRemainingArguments = $false)]
    [string[]]$StudyRoot = @(),
    [string]$Model = "",
    [switch]$SkipMarketplace,
    [switch]$NoBackgroundIndexer
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

function Write-Step([string]$Message) {
    Write-Host "[Study Retriever] $Message"
}

function Assert-PluginPackage([string]$Root) {
    $portableManifest = Join-Path $Root "plugin.json"
    $portableMcp = Join-Path $Root "mcp.json"
    $compatManifest = Join-Path $Root ".codex-plugin\plugin.json"
    $compatMcp = Join-Path $Root ".mcp.json"
    foreach ($path in @($portableManifest, $portableMcp, $compatManifest, $compatMcp)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Plugin package file missing: $path" }
    }
    try {
        $manifest = Get-Content -Raw -LiteralPath $portableManifest | ConvertFrom-Json
        $mcp = Get-Content -Raw -LiteralPath $portableMcp | ConvertFrom-Json
        $compat = Get-Content -Raw -LiteralPath $compatMcp | ConvertFrom-Json
    } catch {
        throw "Plugin package JSON is invalid: $($_.Exception.Message)"
    }
    if ($manifest.'$schema' -ne 'https://agent-plugins.org/schemas/1.0.0/plugin.schema.json') {
        throw "Portable plugin.json has unsupported or missing Agent Plugins schema"
    }
    if ($mcp.'$schema' -ne 'https://agent-plugins.org/schemas/1.0.0/mcp.schema.json') {
        throw "Portable mcp.json has unsupported or missing Agent Plugins schema"
    }
    $server = $mcp.mcpServers.study_retriever
    if (-not $server -or $server.type -ne 'stdio' -or $server.command -ne 'powershell.exe') {
        throw "Portable Study Retriever MCP declaration is invalid"
    }
    $cwd = [string]$server.cwd
    if ($cwd -ne '${PLUGIN_ROOT}' -and -not $cwd.StartsWith('./') -and -not $cwd.StartsWith('${PLUGIN_ROOT}/') -and -not $cwd.StartsWith('${PLUGIN_DATA}/')) {
        throw "Portable MCP cwd is invalid for Agent Plugins: $cwd"
    }
    if (@($server.args).Count -eq 0 -or [string]$server.args[-1] -ne '${PLUGIN_ROOT}/scripts/mcp-launch.ps1') {
        throw "Portable MCP launch script must use the PLUGIN_ROOT path"
    }
    $compatServer = $compat.mcpServers.study_retriever
    if (-not $compatServer -or $compatServer.command -ne 'powershell.exe') {
        throw "Codex compatibility MCP declaration is invalid"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Root 'scripts\mcp-launch.ps1') -PathType Leaf)) {
        throw "MCP launch script is missing from plugin package"
    }
}

function Test-CompatiblePython {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$Prefix = @()
    )

    # A missing runtime behind py.exe writes to stderr. Windows PowerShell 5.1
    # can promote that redirected stderr to a terminating NativeCommandError
    # when the installer uses ErrorActionPreference=Stop. Treat probes as probes.
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "SilentlyContinue"
        & $Command @Prefix -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,14) else 1)" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldPreference
    }
}

function Find-CompatiblePython {
    # Prefer a registered launcher runtime. Missing launcher versions are not errors.
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($version in @("-3.13", "-3.12", "-3.11")) {
            if (Test-CompatiblePython -Command $py.Source -Prefix @($version)) {
                return @{ Command = $py.Source; Prefix = @($version) }
            }
        }
    }

    # Probe every Python visible on PATH. This handles multiple installs, aliases,
    # application-bundled Pythons, and stale PATH entries without assuming order.
    $seen = @{}
    foreach ($name in @("python.exe", "python3.exe", "python3.13.exe", "python3.12.exe", "python3.11.exe")) {
        foreach ($python in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
            if (-not $python.Source) { continue }
            $candidatePath = [System.IO.Path]::GetFullPath($python.Source)
            if ($seen.ContainsKey($candidatePath)) { continue }
            $seen[$candidatePath] = $true
            if ($candidatePath -like "*\WindowsApps\*") { continue }
            if (Test-CompatiblePython -Command $candidatePath) {
                return @{ Command = $candidatePath; Prefix = @() }
            }
        }
    }

    # Probe common per-user installs even when PATH has not been refreshed.
    $localPythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
    if (Test-Path $localPythonRoot) {
        $candidates = Get-ChildItem -Path $localPythonRoot -Filter python.exe -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "Python31[123]" } |
            Sort-Object FullName -Descending
        foreach ($candidate in $candidates) {
            if (Test-CompatiblePython -Command $candidate.FullName) {
                return @{ Command = $candidate.FullName; Prefix = @() }
            }
        }
    }
    return $null
}

function Install-PythonFallback {
    Write-Step "No compatible Python found. Installing Python 3.13.16 for the current user."
    $arch = $env:PROCESSOR_ARCHITECTURE.ToUpperInvariant()
    if ($arch -eq "ARM64") {
        $url = "https://www.python.org/ftp/python/3.13.16/python-3.13.16-arm64.exe"
        $expectedSha256 = "696e2226062c6ec3622c13f143d28336a856968518e49cb4f4c62513215a4f5d"
    } elseif ($arch -in @("AMD64", "X86")) {
        if ($arch -eq "X86") { throw "32-bit Windows is not supported. Use 64-bit Windows 10/11." }
        $url = "https://www.python.org/ftp/python/3.13.16/python-3.13.16-amd64.exe"
        $expectedSha256 = "fb4f9f5d438b2396da0086dc70b935c530cb578e37adc6d354f7ad2037fee83b"
    } else {
        throw "Unsupported Windows architecture: $arch"
    }

    $installer = Join-Path $env:TEMP "study-retriever-python-3.13.16.exe"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $installer
        $actualSha256 = (Get-FileHash -Algorithm SHA256 -Path $installer).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $expectedSha256) {
            throw "Python installer SHA-256 mismatch; refusing to execute downloaded file."
        }
        $target = Join-Path $env:LOCALAPPDATA "Programs\\Python\\Python313"
        $process = Start-Process -FilePath $installer -ArgumentList @(
            "/quiet",
            "InstallAllUsers=0",
            "TargetDir=`"$target`"",
            "PrependPath=0",
            "Include_launcher=1",
            "InstallLauncherAllUsers=0",
            "Include_pip=1",
            "Include_test=0"
        ) -Wait -PassThru
        if ($process.ExitCode -notin @(0, 3010)) {
            throw "Python installer failed with exit code $($process.ExitCode)"
        }
    } finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $installer
    }
}

function Resolve-Python {
    $found = Find-CompatiblePython
    if ($found) { return $found }

    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Step "No compatible Python found. Trying Windows Package Manager (Python 3.13)."
        & $winget.Source install --id Python.Python.3.13 --exact --scope user --silent --accept-package-agreements --accept-source-agreements --disable-interactivity
        if ($LASTEXITCODE -eq 0) {
            $found = Find-CompatiblePython
            if ($found) { return $found }
        }
        Write-Step "winget did not produce a usable Python; using the verified python.org installer fallback."
    }

    Install-PythonFallback
    $found = Find-CompatiblePython
    if ($found) { return $found }
    throw "Python 3.13 installation completed but no compatible interpreter could be located."
}

$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
foreach ($root in $StudyRoot) {
    if (-not (Test-Path -LiteralPath $root)) { throw "Study root does not exist: $root" }
}
$personalMarketplaceDir = Join-Path $HOME ".agents\plugins"
$personalPluginDir = Join-Path $HOME ".codex\plugins"
$pluginRoot = Join-Path $personalPluginDir "study-retriever"
$dataRoot = Join-Path $env:LOCALAPPDATA "StudyRetriever"
$runtimeRoot = Join-Path $dataRoot "runtime"
$venv = Join-Path $runtimeRoot ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $personalMarketplaceDir, $personalPluginDir, $runtimeRoot, $dataRoot | Out-Null

if (Test-Path $venvPython) {
    if (-not (Test-CompatiblePython -Command $venvPython)) {
        Write-Step "Existing isolated runtime is invalid or incompatible; recreating it"
        Remove-Item -Recurse -Force $venv
    }
}

if ([System.IO.Path]::GetFullPath($sourceRoot).TrimEnd('\') -ne [System.IO.Path]::GetFullPath($pluginRoot).TrimEnd('\')) {
    Write-Step "Installing plugin files to $pluginRoot"
    New-Item -ItemType Directory -Force -Path $pluginRoot | Out-Null
    $null = & robocopy.exe $sourceRoot $pluginRoot /MIR /R:2 /W:1 /NFL /NDL /NJH /NJS /NP /XD .git .venv .study-data __pycache__ .pytest_cache build dist /XF *.pyc
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Step "Plugin files already in personal plugin directory"
}

Write-Step "Validating exact Desktop plugin package"
Assert-PluginPackage $pluginRoot

$launcher = Resolve-Python
if (-not (Test-Path $venvPython)) {
    Write-Step "Creating isolated Python runtime at $venv"
    if ($launcher.Prefix.Count -gt 0) {
        & $launcher.Command $launcher.Prefix[0] -m venv $venv
    } else {
        & $launcher.Command -m venv $venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Python virtual environment" }
}

Write-Step "Installing Study Retriever and runtime dependencies"
& $venvPython -m pip install --disable-pip-version-check --retries 5 --timeout 60 --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
$releaseMetadataPath = Join-Path $pluginRoot "release.json"
if (-not (Test-Path -LiteralPath $releaseMetadataPath -PathType Leaf)) {
    throw "release.json is missing from the installed plugin; refusing an unverifiable install."
}
try {
    $releaseMetadata = Get-Content -Raw -LiteralPath $releaseMetadataPath | ConvertFrom-Json
} catch {
    throw "release.json is invalid JSON: $($_.Exception.Message)"
}
if ($releaseMetadata.name -ne "study-retriever" -or -not $releaseMetadata.version) {
    throw "release.json identity/version is invalid"
}
foreach ($manifestPath in @((Join-Path $pluginRoot "plugin.json"), (Join-Path $pluginRoot ".codex-plugin\plugin.json"))) {
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { throw "Plugin manifest missing: $manifestPath" }
    $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
    if ([string]$manifest.version -ne [string]$releaseMetadata.version) {
        throw "Plugin manifest version $($manifest.version) does not match release version $($releaseMetadata.version): $manifestPath"
    }
}
if (-not $releaseMetadata.wheel -or -not $releaseMetadata.wheel.file -or -not $releaseMetadata.wheel.sha256) {
    throw "release.json does not declare the bundled wheel and SHA-256"
}
$bundledWheelPath = [IO.Path]::GetFullPath((Join-Path $pluginRoot ([string]$releaseMetadata.wheel.file)))
$pluginRootFull = [IO.Path]::GetFullPath($pluginRoot).TrimEnd('\') + '\'
if (-not $bundledWheelPath.StartsWith($pluginRootFull, [StringComparison]::OrdinalIgnoreCase)) {
    throw "release.json wheel path escapes the plugin directory"
}
if (-not (Test-Path -LiteralPath $bundledWheelPath -PathType Leaf)) {
    throw "Bundled Study Retriever wheel not found: $bundledWheelPath"
}
$actualBundledWheelSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $bundledWheelPath).Hash.ToLowerInvariant()
$expectedBundledWheelSha256 = ([string]$releaseMetadata.wheel.sha256).ToLowerInvariant()
if ($actualBundledWheelSha256 -ne $expectedBundledWheelSha256) {
    throw "Bundled Study Retriever wheel SHA-256 mismatch; release files may be corrupted or modified."
}
Write-Step "Using verified bundled Study Retriever wheel: $([IO.Path]::GetFileName($bundledWheelPath))"
$installTarget = $bundledWheelPath
& $venvPython -m pip install --disable-pip-version-check --retries 5 --timeout 60 --prefer-binary --no-build-isolation --upgrade $installTarget
if ($LASTEXITCODE -ne 0) { throw "Study Retriever dependency installation failed" }
$env:STUDY_RETRIEVER_EXPECTED_VERSION = [string]$releaseMetadata.version
& $venvPython -c "import os,sys,study_retriever; sys.exit(0 if study_retriever.__version__ == os.environ['STUDY_RETRIEVER_EXPECTED_VERSION'] else 1)"
if ($LASTEXITCODE -ne 0) { throw "Installed Study Retriever wheel version does not match release.json" }
Remove-Item Env:STUDY_RETRIEVER_EXPECTED_VERSION -ErrorAction SilentlyContinue

$env:STUDY_RETRIEVER_HOME = $dataRoot
$env:HF_HUB_ETAG_TIMEOUT = "10"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "60"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"

if ($Model) {
    Write-Step "Selecting embedding model: $Model"
    & $venvPython -m study_retriever.cli set-model $Model
    if ($LASTEXITCODE -ne 0) { throw "Failed to set embedding model" }
    & $venvPython -m study_retriever.cli rebuild-vectors
    if ($LASTEXITCODE -ne 0) { throw "Failed to rebuild vectors for selected model" }
}

Write-Step "Downloading/warming the local embedding model and running a live end-to-end retrieval self-test"
& $venvPython -m study_retriever.cli doctor --full
if ($LASTEXITCODE -ne 0) { throw "Runtime/model/MCP live health check failed" }

foreach ($root in $StudyRoot) {
    Write-Step "Adding and indexing $root"
    & $venvPython -m study_retriever.cli add-root $root
    if ($LASTEXITCODE -ne 0) { throw "Indexing failed for: $root" }
}

if (-not $SkipMarketplace) {
    $marketFile = Join-Path $personalMarketplaceDir "marketplace.json"
    if (Test-Path $marketFile) {
        try {
            $market = Get-Content -Raw -Path $marketFile | ConvertFrom-Json
        } catch {
            throw "Existing marketplace.json is invalid JSON; refusing to overwrite it: $marketFile"
        }
        if (-not ($market.PSObject.Properties.Name -contains "name")) {
            $market | Add-Member -NotePropertyName name -NotePropertyValue "personal-local"
        }
        if (-not ($market.PSObject.Properties.Name -contains "interface")) {
            $market | Add-Member -NotePropertyName interface -NotePropertyValue ([pscustomobject]@{ displayName = "Personal Local Plugins" })
        }
        if (-not ($market.PSObject.Properties.Name -contains "plugins")) {
            $market | Add-Member -NotePropertyName plugins -NotePropertyValue @()
        }
    } else {
        $market = [pscustomobject]@{
            name = "personal-local"
            interface = [pscustomobject]@{ displayName = "Personal Local Plugins" }
            plugins = @()
        }
    }

    $entry = [pscustomobject]@{
        name = "study-retriever"
        source = [pscustomobject]@{ source = "local"; path = "./.codex/plugins/study-retriever" }
        policy = [pscustomobject]@{ installation = "INSTALLED_BY_DEFAULT"; authentication = "ON_INSTALL" }
        category = "Productivity"
    }
    $kept = @($market.plugins | Where-Object { $_.name -ne "study-retriever" })
    $market.plugins = @($kept + $entry)
    $json = $market | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($marketFile, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    Write-Step "Registered personal ChatGPT Desktop marketplace at $marketFile"
}

# ChatGPT Desktop executes a cached copy of local-marketplace plugins. Clearing only
# this plugin's disposable cache guarantees the next Desktop restart reloads the
# freshly installed source while preserving every other plugin and all study data.
$pluginCacheRoot = Join-Path $HOME ".codex\plugins\cache"
if (Test-Path -LiteralPath $pluginCacheRoot -PathType Container) {
    foreach ($marketCache in @(Get-ChildItem -LiteralPath $pluginCacheRoot -Directory -ErrorAction SilentlyContinue)) {
        $pluginCache = Join-Path $marketCache.FullName "study-retriever"
        if (Test-Path -LiteralPath $pluginCache) {
            Write-Step "Clearing stale ChatGPT local-plugin cache: $pluginCache"
            try {
                Remove-Item -Recurse -Force -LiteralPath $pluginCache -ErrorAction Stop
            } catch {
                # The active Codex/Desktop host can keep its current cache version open.
                # Source/runtime are already installed and health-checked; defer cache
                # cleanup until that host exits instead of reporting a false install failure.
                Write-Step "Cache is in use; leaving the active host untouched. Restart the host, then rerun install to clear stale cache: $pluginCache"
            }
        }
    }
}

Write-Step "Final index consistency check"
& $venvPython -m study_retriever.cli status
if ($LASTEXITCODE -ne 0) { throw "Final status check failed" }

if (-not $NoBackgroundIndexer) {
    $schtasks = Get-Command schtasks.exe -ErrorAction SilentlyContinue
    $indexerScript = Join-Path $pluginRoot "scripts\indexer-launch.ps1"
    if ($schtasks -and (Test-Path -LiteralPath $indexerScript -PathType Leaf)) {
        $taskName = "Study Retriever Background Indexer"
        $taskAction = "powershell.exe -NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$indexerScript`""
        Write-Step "Registering continuous background indexing at user logon"
        & $schtasks.Source /Create /TN $taskName /SC ONLOGON /TR $taskAction /RL LIMITED /F | Out-Null
        if ($LASTEXITCODE -eq 0) {
            & $schtasks.Source /Run /TN $taskName | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Step "Background task is registered but could not be started immediately; it will start at next logon."
            }
        } else {
            # Task Scheduler can refuse non-elevated registration; a per-user Run key needs no admin rights.
            try {
                Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "StudyRetrieverIndexer" -Value $taskAction
                Write-Step "Task Scheduler refused registration; registered the background indexer under HKCU Run (starts at next logon)."
            } catch {
                Write-Step "Could not register the optional background indexer. Live indexing still runs through the ChatGPT MCP process."
            }
        }
    } else {
        Write-Step "Task Scheduler unavailable; live indexing will run through the ChatGPT MCP process only."
    }
}

Write-Host ""
Write-Host "Installed and live-tested. Restart ChatGPT Desktop and open a NEW Work or Codex chat for the guaranteed local-marketplace surface. If the plugin UI asks for confirmation, enable Study Retriever once under Plugins > Personal Local Plugins."
if ($StudyRoot.Count -eq 0) {
    Write-Host "No new study root was supplied. Existing configured roots were preserved. If none exist, rerun:"
    Write-Host ".\scripts\install.ps1 -StudyRoot 'D:\path\to\study-material'"
}
