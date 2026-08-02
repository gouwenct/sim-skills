[CmdletBinding()]
param(
    [string]$TargetRoot = (Join-Path $env:USERPROFILE '.agents\skills')
)

$ErrorActionPreference = 'Stop'

$skillsRoot = Join-Path $PSScriptRoot 'skills'
$skillNames = @(
    'r-3dmpsim',
    'r-mfsim',
    'mp-data-ingest',
    'mp-preprocess',
    'simcheck',
    'mp-reconstruct',
    'mf-reconstruct',
    'raw-check'
)

New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null

foreach ($skillName in $skillNames) {
    $source = (Get-Item -LiteralPath (Join-Path $skillsRoot $skillName)).FullName
    $target = Join-Path $TargetRoot $skillName

    if (Test-Path -LiteralPath $target) {
        $existing = Get-Item -Force -LiteralPath $target
        $existingTargets = @($existing.Target) | ForEach-Object {
            if ($_ -ne $null) { [IO.Path]::GetFullPath([string]$_) }
        }

        if ($existing.LinkType -eq 'Junction' -and $existingTargets -contains $source) {
            Write-Host "Already linked: $skillName"
            continue
        }

        throw "Refusing to replace existing path: $target"
    }

    New-Item -ItemType Junction -Path $target -Target $source | Out-Null
    Write-Host "Linked: $skillName"
}
