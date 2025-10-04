Param()

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir '..') | Select-Object -ExpandProperty Path
$outputDir = Join-Path $scriptDir 'out'
$doctorLog = Join-Path $outputDir 'doctor_imports.txt'

# Ensure output directory exists
if (-not (Test-Path $outputDir)) {
    New-Item -Path $outputDir -ItemType Directory | Out-Null
}

Write-Host "[doctor] repo_root=$repoRoot"

# Remove bytecode artefacts
Write-Host '[doctor] purging __pycache__ and *.pyc'
Get-ChildItem -Path $repoRoot -Recurse -Force -Filter '__pycache__' -Directory | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
}
Get-ChildItem -Path $repoRoot -Recurse -Force -Include '*.pyc' -File | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
}

# Ensure only one project clone is importable within parent directory
$parentDir = Split-Path -Parent $repoRoot
$candidateDirs = Get-ChildItem -Path $parentDir -Directory | Where-Object { $_.Name -like 'video_pipeline*' }
$importable = @()
foreach ($item in $candidateDirs) {
    $candidatePath = Join-Path $item.FullName 'pipeline_core\llm_service.py'
    if (Test-Path $candidatePath) {
        $importable += $item.FullName
    }
}
if ($importable.Count -gt 1) {
    throw "Multiple project directories detected: $($importable -join ', ')"
}

# Preserve and override environment variables for clean run
$prevDontWrite = $env:PYTHONDONTWRITEBYTECODE
$prevWarnings = $env:PYTHONWARNINGS
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONWARNINGS = 'default'

try {
    Push-Location $repoRoot
    Write-Host '[doctor] running pipeline import diagnostics'
    $commandOutput = & python -B -X importtime run_pipeline.py --diag-broll --no-report 2>&1
    $commandOutput | Tee-Object -FilePath $doctorLog
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        Write-Warning "run_pipeline.py exited with code $exitCode"
    }
} finally {
    Pop-Location
    if ($null -eq $prevDontWrite) { Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue } else { $env:PYTHONDONTWRITEBYTECODE = $prevDontWrite }
    if ($null -eq $prevWarnings) { Remove-Item Env:PYTHONWARNINGS -ErrorAction SilentlyContinue } else { $env:PYTHONWARNINGS = $prevWarnings }
}

if (-not (Test-Path $doctorLog)) {
    throw 'doctor_imports.txt was not created'
}

# Validate module hashes
$moduleLines = Select-String -Path $doctorLog -Pattern '\[module pipeline_core\\.llm_service\]' -SimpleMatch
if ($moduleLines) {
    $hashes = @{}
    foreach ($line in $moduleLines) {
        if ($line.Line -match 'sha256=([0-9a-fA-F<>]+)') {
            $hash = $Matches[1]
            $hashes[$hash] = $true
        }
    }
    if ($hashes.Keys.Count -gt 1) {
        throw "Conflicting pipeline_core.llm_service sha256 values detected: $($hashes.Keys -join ', ')"
    }
}
else {
    Write-Warning 'No module stamp found for pipeline_core.llm_service; check runtime banner output.'
}

Write-Host "[doctor] diagnostics captured at $doctorLog"
