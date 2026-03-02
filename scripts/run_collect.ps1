<#
.SYNOPSIS
    Lance collect_status.py en arriere-plan, sans mise en veille, avec journalisation.

.DESCRIPTION
    - Empeche Windows de mettre l'ordinateur en veille (via --keep-awake dans Python).
    - Lance Python en arriere-plan (processus detache, fenetre cachee).
    - Redirige stdout + stderr vers logs/collect_YYYY-MM-DD_HH-mm.log.
    - Affiche le PID et les commandes pour surveiller ou arreter la collecte.

.PARAMETER Interval
    Intervalle entre snapshots en secondes (defaut : 60).

.PARAMETER Duration
    Duree totale de collecte en secondes. 0 = infini (defaut).

.PARAMETER Systems
    Liste de system_id a collecter. Defaut = toutes les villes prioritaires.
    Exemple : -Systems Paris,lyon,toulouse

.PARAMETER Workers
    Nombre de threads HTTP paralleles (defaut : 12).

.PARAMETER MaxIter
    Nombre maximal d'iterations (0 = pas de limite).

.PARAMETER Foreground
    Si present, affiche la sortie dans la console (mode debug/test).

.EXAMPLE
    # Collecte continue sur toutes les villes (8h)
    .\scripts\run_collect.ps1 -Duration 28800

    # Test rapide : 5 snapshots sur Paris et Lyon
    .\scripts\run_collect.ps1 -Systems Paris,lyon -Interval 30 -MaxIter 5 -Foreground

    # Collecte nuit entiere avec 16 workers
    .\scripts\run_collect.ps1 -Duration 36000 -Workers 16
#>

param(
    [int]      $Interval   = 60,
    [int]      $Duration   = 0,
    [string[]] $Systems    = @(),
    [int]      $Workers    = 12,
    [int]      $MaxIter    = 0,
    [switch]   $Foreground
)

$ErrorActionPreference = "Continue"

# Racine du projet (dossier parent de /scripts)
$Root    = Split-Path $PSScriptRoot -Parent
$LogDir  = Join-Path $Root "logs"
$PidFile = Join-Path $LogDir "collect_current.pid"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$LogFile   = Join-Path $LogDir "collect_$Timestamp.log"
$ErrFile   = Join-Path $LogDir "collect_${Timestamp}_err.log"

# Construction des arguments Python
$Script = Join-Path $Root "scripts\collect_status.py"
$PyArgs = @($Script, "--keep-awake", "--workers", $Workers, "--interval", $Interval)

if ($Duration -gt 0) {
    $PyArgs += @("--duration", $Duration)
}
if ($MaxIter -gt 0) {
    $PyArgs += @("--max-iter", $MaxIter)
}
if ($Systems.Count -gt 0) {
    $PyArgs += @("--systems") + $Systems
}

# Calcul des chaines d'affichage (evite les subexpressions imbriquees)
if ($Duration -gt 0) {
    $DurationStr = "$Duration s"
} else {
    $DurationStr = "infinie (Ctrl+C ou Stop-Process pour arreter)"
}

if ($Systems.Count -gt 0) {
    $VillesStr = $Systems -join ", "
} else {
    $VillesStr = "toutes les prioritaires (~42 villes)"
}

Write-Host ""
Write-Host "=== Collecte GBFS VLS - Lanceur PowerShell ===" -ForegroundColor Cyan
Write-Host "Script    : $Script"
Write-Host "Workers   : $Workers threads HTTP paralleles"
Write-Host "Intervalle: $Interval s"
Write-Host "Duree     : $DurationStr"
Write-Host "Villes    : $VillesStr"
Write-Host "Log       : $LogFile"
Write-Host ""

# Mode premier plan (debug)
# On force UTF-8 pour eviter les caracteres corrompus dans les logs Python
if ($Foreground) {
    Write-Host "[mode premier plan - Ctrl+C pour arreter]" -ForegroundColor Yellow
    Write-Host ""
    $env:PYTHONIOENCODING = "utf-8"
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    & python @PyArgs 2>&1 | Tee-Object -FilePath $LogFile
    Write-Host ""
    Write-Host "Collecte terminee." -ForegroundColor Green
    exit 0
}

# Mode arriere-plan : Start-Process detache le processus de la console.
# Python gere lui-meme SetThreadExecutionState via --keep-awake.
# Note : Start-Process joint les elements sans guillemets — il faut quoting explicite
# pour les chemins contenant des espaces (ex. "OneDrive - Cesi").
$BgArgs = @("`"$Script`"") + $PyArgs[1..($PyArgs.Length - 1)]
$Proc = Start-Process python `
    -ArgumentList $BgArgs `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError  $ErrFile `
    -WindowStyle Hidden `
    -PassThru

$Proc.Id | Set-Content $PidFile

Write-Host "Collecte demarree en arriere-plan." -ForegroundColor Green
Write-Host "PID    : $($Proc.Id)"
Write-Host "Log    : $LogFile"
Write-Host ""
Write-Host "--- Commandes utiles ---" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "Surveiller en direct :"
Write-Host "  Get-Content '$LogFile' -Wait -Tail 20"
Write-Host ""
Write-Host "Verifier si le processus tourne encore :"
Write-Host "  Get-Process -Id $($Proc.Id) -ErrorAction SilentlyContinue"
Write-Host ""
Write-Host "Arreter la collecte :"
Write-Host "  Stop-Process -Id $($Proc.Id)"
Write-Host ""
Write-Host "Consulter les erreurs eventuelles :"
Write-Host "  Get-Content '$ErrFile'"
Write-Host ""
