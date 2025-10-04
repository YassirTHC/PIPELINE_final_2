Param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$doctorPath = Join-Path $scriptDir 'doctor.py'

if (-not (Test-Path $doctorPath)) {
    throw "doctor.py not found at $doctorPath"
}

$python = $env:PYTHON
if (-not $python -and $env:VIRTUAL_ENV) {
    $candidate = Join-Path $env:VIRTUAL_ENV 'Scripts/python.exe'
    if (Test-Path $candidate) {
        $python = $candidate
    }
}
if (-not $python) {
    $python = 'python'
}

& $python $doctorPath @Args
exit $LASTEXITCODE
