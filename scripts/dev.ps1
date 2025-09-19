param(
    [string]$Video = "clips/demo.mp4",
    [switch]$Legacy,
    [switch]$NoVerbose
)

$repo = (Resolve-Path "$PSScriptRoot/..\").Path
$env:PYTHONPATH = "$repo;$repo\AI-B-roll;$repo\utils"
$env:PYTHONIOENCODING = "utf-8"
if ($Legacy) {
    $env:ENABLE_PIPELINE_CORE_FETCHER = "false"
} elseif (-not $env:ENABLE_PIPELINE_CORE_FETCHER) {
    $env:ENABLE_PIPELINE_CORE_FETCHER = "true"
}

$videoCandidate = Join-Path $repo $Video
if (-not (Test-Path $videoCandidate)) {
    throw "Could not locate target video: $Video"
}
$videoPath = (Resolve-Path $videoCandidate).Path

$cmd = @("$repo\run_pipeline.py", "--video", $videoPath)
if (-not $NoVerbose) { $cmd += "--verbose" }
if ($Legacy) { $cmd += "--legacy" }

python @cmd