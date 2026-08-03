[CmdletBinding()]
param(
    [ValidateSet('core', 'fwhm', 'decorr')]
    [string]$Profile = 'core',
    [string]$Python,
    [switch]$CreateUv
)

$ErrorActionPreference = 'Stop'
$resolver = Join-Path $PSScriptRoot 'tools\python_env.py'
$resolutionError = 'No bootstrap Python is available to inspect existing environments.'

if ($Python) {
    $env:SIM_SKILLS_PYTHON = $Python
}

$bootstrap = Get-Command python -ErrorAction SilentlyContinue
if ($bootstrap) {
    $resolved = & $bootstrap.Source -B $resolver $Profile 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Output $resolved
        exit 0
    }
    $resolutionError = ($resolved | Out-String).Trim()
    if ($resolutionError -notlike 'No compatible*') {
        throw $resolutionError
    }
}

if (-not $CreateUv) {
    throw "$resolutionError Ask the user before rerunning with -CreateUv."
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw 'uv is required to create the shared CPU-only environment.'
}

& $uv.Source sync --project $PSScriptRoot --no-dev
if ($LASTEXITCODE -ne 0) {
    throw 'uv sync failed.'
}

$sharedPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$env:SIM_SKILLS_PYTHON = $sharedPython
& $sharedPython -B $resolver $Profile
