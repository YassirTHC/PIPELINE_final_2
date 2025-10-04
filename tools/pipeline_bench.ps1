[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Models,
    [string]$Video = 'clips/121.mp4',
    [int]$AllowImages = 0,
    [int]$RetriesOnEmpty = 1,
    [string]$Providers = 'pexels,pixabay',
    [switch]$NudgeOnEmpty
)

$ErrorActionPreference = 'Stop'

function Get-RegexCount {
    param(
        [string]$Text,
        [string]$Pattern,
        [switch]$CaseSensitive
    )
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return 0
    }
    $options = [System.Text.RegularExpressions.RegexOptions]::Multiline
    if (-not $CaseSensitive) {
        $options = $options -bor [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
    }
    return [System.Text.RegularExpressions.Regex]::Matches($Text, $Pattern, $options).Count
}

function Get-FillRateFromLog {
    param(
        [string]$LogText
    )
    if (-not $LogText) {
        return $null
    }
    $pattern = 'B-roll sélectionnés:\s*(?<selected>\d+)/(\s*(?<target>\d+))'
    $match = [System.Text.RegularExpressions.Regex]::Match($LogText, $pattern)
    if ($match.Success) {
        $selected = [int]$match.Groups['selected'].Value
        $target = [int]$match.Groups['target'].Value
        return [pscustomobject]@{
            selected = $selected
            target = $target
            fill_rate = if ($target -gt 0) { [math]::Round($selected / $target, 4) } else { $null }
            source = 'log'
        }
    }
    return $null
}

function Get-FillRateFromReport {
    param(
        [string[]]$CandidatePaths
    )
    foreach ($path in $CandidatePaths) {
        if (-not $path) { continue }
        if (-not (Test-Path $path)) { continue }
        try {
            $content = Get-Content -Path $path -Raw
            if ([string]::IsNullOrWhiteSpace($content)) { continue }
            $json = $content | ConvertFrom-Json
        }
        catch {
            continue
        }
        $selected = $null
        $target = $null
        if ($json.PSObject.Properties.Name -contains 'selected' -and $json.PSObject.Properties.Name -contains 'target') {
            $selected = [int]$json.selected
            $target = [int]$json.target
        }
        elseif ($json.broll -and $json.broll.selected -ne $null) {
            $selected = [int]$json.broll.selected
            $target = [int]$json.broll.target
        }
        if ($selected -ne $null -and $target -ne $null) {
            return [pscustomobject]@{
                selected = $selected
                target = $target
                fill_rate = if ($target -gt 0) { [math]::Round($selected / $target, 4) } else { $null }
                source = $path
            }
        }
    }
    return $null
}

function Get-BrollFilterStats {
    param(
        [string]$EventsPath
    )
    $result = [pscustomobject]@{
        duration_drops = 0.0
        orientation_drops = 0.0
        event_count = 0
        source = $EventsPath
        error = $null
    }
    if (-not (Test-Path $EventsPath)) {
        $result.error = 'missing'
        return $result
    }
    try {
        foreach ($line in Get-Content -Path $EventsPath) {
            if ([string]::IsNullOrWhiteSpace($line)) { continue }
            try {
                $jsonLine = $line | ConvertFrom-Json
            }
            catch {
                continue
            }
            if ($jsonLine.event -ne 'broll_filter_stats') { continue }
            $result.event_count++
            if ($jsonLine.PSObject.Properties.Name -contains 'filter_duration' -and $jsonLine.filter_duration -ne $null) {
                try { $result.duration_drops += [double]$jsonLine.filter_duration } catch {}
            }
            if ($jsonLine.PSObject.Properties.Name -contains 'filter_orientation' -and $jsonLine.filter_orientation -ne $null) {
                try { $result.orientation_drops += [double]$jsonLine.filter_orientation } catch {}
            }
        }
    }
    catch {
        $result.error = $_.Exception.Message
    }
    $result.duration_drops = [math]::Round($result.duration_drops, 3)
    $result.orientation_drops = [math]::Round($result.orientation_drops, 3)
    return $result
}

function Get-DiagBrollSummary {
    param(
        [string]$DiagPath
    )
    $summary = [pscustomobject]@{
        source_path = $DiagPath
        providers = @()
        success_count = 0
        failure_count = 0
        candidates_total = 0
        latency_avg_sec = $null
        no_results = 0
        timeouts = 0
        error = $null
    }
    if (-not (Test-Path $DiagPath)) {
        $summary.error = 'missing'
        return $summary
    }
    try {
        $raw = Get-Content -Path $DiagPath -Raw
        if ([string]::IsNullOrWhiteSpace($raw)) {
            $summary.error = 'empty_file'
            return $summary
        }
        $json = $raw | ConvertFrom-Json
    }
    catch {
        $summary.error = 'parse_error'
        return $summary
    }
    $providerSummaries = @()
    $latencies = @()
    if ($json.providers) {
        foreach ($provider in $json.providers) {
            $item = [pscustomobject]@{
                name = $provider.name
                success = [bool]$provider.success
                candidates = $provider.candidates
                latency_sec = $provider.latency
                error = $provider.error
            }
            if ($provider.success) {
                $summary.success_count++
            }
            else {
                $summary.failure_count++
            }
            if ($provider.candidates -ne $null) {
                try { $summary.candidates_total += [int]$provider.candidates } catch {}
            }
            if ($provider.latency -ne $null) {
                try { $latencies += [double]$provider.latency } catch {}
            }
            $providerSummaries += $item
        }
    }
    if ($latencies.Count -gt 0) {
        $summary.latency_avg_sec = [math]::Round(($latencies | Measure-Object -Average).Average, 3)
    }
    if ($json.no_results -ne $null) {
        $summary.no_results = [int]$json.no_results
    }
    if ($json.timeout -ne $null) {
        $summary.timeouts = [int]$json.timeout
    }
    if ($json.stats) {
        if ($json.stats.no_results -ne $null) {
            $summary.no_results = [int]$json.stats.no_results
        }
        if ($json.stats.timeout -ne $null) {
            $summary.timeouts = [int]$json.stats.timeout
        }
    }
    $summary.providers = $providerSummaries
    return $summary
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Data,
        [int]$Depth = 6
    )
    $jsonContent = $Data | ConvertTo-Json -Depth $Depth
    Set-Content -Path $Path -Value $jsonContent -Encoding UTF8
}

function Write-Markdown {
    param(
        [string]$Path,
        [string[]]$Lines
    )
    Set-Content -Path $Path -Value ($Lines -join "`n") -Encoding UTF8
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
if (-not $repoRoot) {
    $repoRoot = Get-Location
}

chcp.com 65001 | Out-Null
$env:PYTHONIOENCODING = 'utf-8'

$env:PIPELINE_LLM_ENDPOINT = 'http://localhost:11434'
$env:BROLL_FETCH_FORCE_IPV4 = '1'
$env:BROLL_FORCE_IPV4 = '1'
$env:BROLL_FETCH_PROVIDER = $Providers
$env:BROLL_FETCH_ALLOW_VIDEOS = '1'
$env:BROLL_FETCH_ALLOW_IMAGES = [string]$AllowImages
$env:BROLL_FETCH_REQUEST_TIMEOUT_S = '12'
$env:BROLL_FETCH_SEGMENT_TIMEOUT_S = '20'

$modelsList = $Models -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($modelsList.Count -eq 0) {
    throw 'No models provided.'
}

$outputDir = Join-Path $scriptRoot 'out'
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$pipelinePath = Join-Path $repoRoot 'run_pipeline.py'
if (-not (Test-Path $pipelinePath)) {
    throw "Cannot locate run_pipeline.py at $pipelinePath"
}

$benchMatrix = @()

Remove-Item Env:PIPELINE_LLM_PROMPT_NUDGE -ErrorAction SilentlyContinue

foreach ($model in $modelsList) {
    if ([string]::IsNullOrWhiteSpace($model)) { continue }
    Write-Host "--- Benchmarking $model ---"

    $env:PIPELINE_LLM_MODEL = $model
    $env:PIPELINE_LLM_TIMEOUT_S = '90'
    $env:PIPELINE_LLM_NUM_PREDICT = '192'
    $env:PIPELINE_LLM_TEMP = '0.2'
    $env:PIPELINE_LLM_TOP_P = '0.85'
    $env:PIPELINE_LLM_JSON_MODE = '1'

    $safeName = ($model -replace '[^a-zA-Z0-9_\.-]', '_')

    $diagStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $diagOutput = & python $pipelinePath --diag-broll --no-report 2>&1
    $diagExit = $LASTEXITCODE
    $diagStopwatch.Stop()

    $diagPath = Join-Path $repoRoot 'diagnostic_broll.json'
    $diagSummary = Get-DiagBrollSummary -DiagPath $diagPath
    $copiedDiagPath = Join-Path $outputDir "diagnostic_broll_$($safeName).json"
    if (Test-Path $diagPath) {
        Copy-Item -Path $diagPath -Destination $copiedDiagPath -Force
    }

    $attempt = 1
    $maxAttempts = [math]::Max(1, 1 + [math]::Max(0, $RetriesOnEmpty))
    $emptyPayloadCount = 0
    $finalLogPath = Join-Path $outputDir "pipeline_$($safeName).log"
    $attemptLogPaths = @()
    $pipelineExit = 0
    $pipelineDuration = 0.0

    while ($attempt -le $maxAttempts) {
        if ($attempt -gt 1) {
            if ($NudgeOnEmpty.IsPresent) {
                $env:PIPELINE_LLM_PROMPT_NUDGE = 'Answer in plain English.'
            }
            else {
                Remove-Item Env:PIPELINE_LLM_PROMPT_NUDGE -ErrorAction SilentlyContinue
            }
        }
        $attemptLog = Join-Path $outputDir "pipeline_$($safeName)_attempt$attempt.log"
        $pipelineStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        & python $pipelinePath --video $Video --verbose 2>&1 | Tee-Object -FilePath $attemptLog | Out-Null
        $pipelineExit = $LASTEXITCODE
        $pipelineStopwatch.Stop()
        $pipelineDuration = [math]::Round($pipelineStopwatch.Elapsed.TotalSeconds, 3)

        $attemptLogPaths += $attemptLog
        $logText = if (Test-Path $attemptLog) { Get-Content -Path $attemptLog -Raw } else { '' }
        $emptyPayloadCount = Get-RegexCount -Text $logText -Pattern 'empty response payload'
        if ($emptyPayloadCount -eq 0) {
            Copy-Item -Path $attemptLog -Destination $finalLogPath -Force
            break
        }
        elseif ($attempt -lt $maxAttempts) {
            Write-Host "Retrying due to empty response payload (attempt $attempt)" -ForegroundColor Yellow
        }
        else {
            Copy-Item -Path $attemptLog -Destination $finalLogPath -Force
        }
        $attempt++
    }

    Remove-Item Env:PIPELINE_LLM_PROMPT_NUDGE -ErrorAction SilentlyContinue

    $finalLogText = if (Test-Path $finalLogPath) { Get-Content -Path $finalLogPath -Raw } else { '' }

    $segmentCount = Get-RegexCount -Text $finalLogText -Pattern 'source=llm_segment'
    $metadataCount = Get-RegexCount -Text $finalLogText -Pattern 'source=llm_metadata'

    $logFill = Get-FillRateFromLog -LogText $finalLogText
    if (-not $logFill) {
        $fillPaths = @(
            Join-Path $repoRoot 'selection_report_reframed.json',
            Join-Path $repoRoot 'output/selection_report_reframed.json',
            Join-Path $repoRoot 'output/meta/selection_report_reframed.json'
        )
        $logFill = Get-FillRateFromReport -CandidatePaths $fillPaths
    }

    $selectedCount = if ($logFill) { $logFill.selected } else { $null }
    $targetCount = if ($logFill) { $logFill.target } else { $null }
    $fillRate = if ($logFill -and $logFill.fill_rate -ne $null) { $logFill.fill_rate } else { $null }

    $logLines = @()
    if ($finalLogText) {
        $logLines = $finalLogText -split "`n"
    }
    $brollLines = $logLines | Where-Object { $_ -match 'B-?roll' -or $_ -match '\[BROLL' }
    $brollCorpus = $brollLines -join ' '

    $providerNoisePattern = '\b(visuals?|clips?|video|shots?|scene|sequence|demonstration|visualization|process|realization|mapping)\b'
    $genericPattern = '\b(that|this|it|they|we|you|thing|stuff|very|just|really)\b'
    $providerNoiseCount = Get-RegexCount -Text $brollCorpus -Pattern $providerNoisePattern
    $genericCount = Get-RegexCount -Text $brollCorpus -Pattern $genericPattern

    $eventsPath = Join-Path $repoRoot 'output/meta/broll_pipeline_events.jsonl'
    $filterStats = Get-BrollFilterStats -EventsPath $eventsPath

    $benchObject = [ordered]@{
        model = $model
        safe_name = $safeName
        generated_at = (Get-Date).ToString('o')
        settings = [ordered]@{
            video = $Video
            allow_images = $AllowImages
            providers = $Providers
            retries_on_empty = $RetriesOnEmpty
            nudge_on_empty = [bool]$NudgeOnEmpty.IsPresent
        }
        attempts = [ordered]@{
            total_attempts = $attempt
            attempt_logs = $attemptLogPaths
            exit_codes = [ordered]@{
                diag = $diagExit
                pipeline = $pipelineExit
            }
            duration_sec = [ordered]@{
                diag = [math]::Round($diagStopwatch.Elapsed.TotalSeconds, 3)
                pipeline = $pipelineDuration
            }
        }
        diagnostics = [ordered]@{
            diag_broll = $diagSummary
        }
        metrics = [ordered]@{
            fill = [ordered]@{
                selected = $selectedCount
                target = $targetCount
                fill_rate = $fillRate
                source = if ($logFill) { $logFill.source } else { 'unavailable' }
            }
            llm = [ordered]@{
                empty_payload_count = $emptyPayloadCount
                llm_segment_count = $segmentCount
                llm_metadata_count = $metadataCount
            }
            noise = [ordered]@{
                provider_noise_count = $providerNoiseCount
                generic_terms_count = $genericCount
            }
            filters = [ordered]@{
                duration_drops = $filterStats.duration_drops
                orientation_drops = $filterStats.orientation_drops
                event_count = $filterStats.event_count
                source = $filterStats.source
                error = $filterStats.error
            }
        }
        files = [ordered]@{
            pipeline_log = $finalLogPath
            diagnostic_log = $copiedDiagPath
        }
    }

    $benchJsonPath = Join-Path $outputDir "bench_$($safeName).json"
    Write-JsonFile -Path $benchJsonPath -Data $benchObject

    $mdLines = @(
        "# Pipeline Benchmark — $model",
        '',
        "Generated: $((Get-Date).ToString('u'))",
        '',
        "- Attempts: $($benchObject.attempts.total_attempts) (empty payload count: $emptyPayloadCount)",
        "- Fill rate: $($selectedCount)/$($targetCount) (source: $($benchObject.metrics.fill.source))",
        "- LLM segments: $segmentCount, metadata: $metadataCount",
        "- Filter drops — duration: $($filterStats.duration_drops), orientation: $($filterStats.orientation_drops)",
        "- Noise counts — provider: $providerNoiseCount, generic: $genericCount",
        "- Diag providers success: $($diagSummary.success_count) / candidates total: $($diagSummary.candidates_total)",
        '',
        '## Files',
        "- Pipeline log: $finalLogPath",
        "- Diagnostic copy: $copiedDiagPath"
    )
    $benchMdPath = Join-Path $outputDir "bench_$($safeName).md"
    Write-Markdown -Path $benchMdPath -Lines $mdLines

    $benchMatrix += [pscustomobject]@{
        model = $model
        fill_rate = $fillRate
        fill_selected = $selectedCount
        fill_target = $targetCount
        empty_payload = $emptyPayloadCount
        llm_segments = $segmentCount
        llm_metadata = $metadataCount
        provider_noise = $providerNoiseCount
        generic_terms = $genericCount
        duration_drops = $filterStats.duration_drops
        orientation_drops = $filterStats.orientation_drops
        diag_success = $diagSummary.success_count
        diag_candidates = $diagSummary.candidates_total
        diag_latency_avg = $diagSummary.latency_avg_sec
        attempts = $benchObject.attempts.total_attempts
    }
}

if ($benchMatrix.Count -gt 0) {
    $csvPath = Join-Path $outputDir 'bench_matrix.csv'
    $benchMatrix | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
}
