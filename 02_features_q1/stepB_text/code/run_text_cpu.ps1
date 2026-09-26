param(
    [string]$Workspace = "E:\myProject\postgraduate\Coding\CPMCM\E_workspace",
    [string]$ModelPath = "",
    [int]$BatchSize = 4,
    [int]$MaxLength = 256,
    [switch]$LocalFilesOnly
)

$script = Join-Path $PSScriptRoot "extract_text_features.py"
$validator = Join-Path $PSScriptRoot "validate_text_features.py"
$python = $null
if ($env:CONDA_PREFIX) {
    $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path -LiteralPath $condaPython) { $python = $condaPython }
}
if (-not $python) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

$pythonVersion = & $python -c "import sys; print(sys.version_info.major, sys.version_info.minor)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($pythonVersion.Trim() -ne "3 11") {
    Write-Error "Expected Python 3.11 from the configured CPU environment; got '$pythonVersion' using '$python'. Activate cpmcm-e-cpu first."
    exit 2
}

$cliArgs = @($script, "--workspace", $Workspace, "--device", "cpu", "--batch-size", $BatchSize, "--max-length", $MaxLength)
if ($ModelPath) { $cliArgs += @("--model-path", $ModelPath) }
if ($LocalFilesOnly) { $cliArgs += "--local-files-only" }
& $python @cliArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python $validator --workspace $Workspace
exit $LASTEXITCODE
