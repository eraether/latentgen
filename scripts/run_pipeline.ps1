# Run every stage in order with one config (PowerShell twin of run_pipeline.sh).
#
#   .\scripts\run_pipeline.ps1 configs\ffhq512.yaml
#   .\scripts\run_pipeline.ps1 configs\smoke_test.yaml
#   .\scripts\run_pipeline.ps1 configs\ffhq512.yaml -From 2      # start at stage 2 (encoded data exists)
param([string]$Config = "configs/ffhq512.yaml", [string]$From = "1a")
$ErrorActionPreference = "Stop"
$Config = (Resolve-Path $Config).Path
Set-Location (Join-Path $PSScriptRoot "..")
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$stages = [ordered]@{
    "1a" = "scripts/01a_train_vqvae.py"; "1b" = "scripts/01b_train_autoencoder.py"; "1c" = "scripts/01c_encode_dataset.py"
    "2"  = "scripts/02_train_maskgit.py"; "3"  = "scripts/03_train_cgan.py";         "4"  = "scripts/04_generate.py"
}
$started = $false
foreach ($stage in $stages.Keys) {
    if ($stage -eq $From) { $started = $true }
    if (-not $started) { continue }
    Write-Host "================================================================ stage $stage"
    & $py $stages[$stage] --config $Config
    if ($LASTEXITCODE -ne 0) { throw "stage $stage failed" }
}
Write-Host "pipeline finished"
