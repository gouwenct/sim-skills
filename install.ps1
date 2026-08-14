[CmdletBinding()]
param(
    [string]$TargetRoot = (Join-Path $env:USERPROFILE '.agents\skills'),
    [switch]$MigrateExisting,
    [string]$BackupRoot
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
    'raw-check',
    'raw-fpj',
    'fwhm',
    'decorr',
    'hyperstacks'
)

New-Item -ItemType Directory -Path $TargetRoot -Force | Out-Null
$links = @()
$conflicts = @()

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
        $conflicts += [pscustomobject]@{ Name = $skillName; Path = $target }
        continue
    }
    $links += [pscustomobject]@{ Name = $skillName; Source = $source; Target = $target }
}

if ($conflicts -and -not $MigrateExisting) {
    $paths = ($conflicts.Path | ForEach-Object { "- $_" }) -join [Environment]::NewLine
    throw "Refusing to replace existing paths. Confirm backup migration, then rerun with -MigrateExisting:$([Environment]::NewLine)$paths"
}

if ($conflicts) {
    if (-not $BackupRoot) {
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $BackupRoot = Join-Path (Split-Path $TargetRoot -Parent) "skills-backup\$stamp"
    }
    New-Item -ItemType Directory -Path $BackupRoot -Force | Out-Null
    foreach ($conflict in $conflicts) {
        $backup = Join-Path $BackupRoot $conflict.Name
        if (Test-Path -LiteralPath $backup) {
            throw "Backup target already exists: $backup"
        }
    }
    foreach ($conflict in $conflicts) {
        $backup = Join-Path $BackupRoot $conflict.Name
        Move-Item -LiteralPath $conflict.Path -Destination $backup
        Write-Host "Backed up: $($conflict.Name) -> $backup"
        $source = (Get-Item -LiteralPath (Join-Path $skillsRoot $conflict.Name)).FullName
        $links += [pscustomobject]@{ Name = $conflict.Name; Source = $source; Target = $conflict.Path }
    }
}

foreach ($link in $links) {
    New-Item -ItemType Junction -Path $link.Target -Target $link.Source | Out-Null
    Write-Host "Linked: $($link.Name)"
}
