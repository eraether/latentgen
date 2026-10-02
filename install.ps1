# =============================================================================================
#  latentgen installer (Windows PowerShell). WSL users: use ./install.sh inside WSL instead.
#
#    .\install.ps1            create .venv, install PyTorch for your NVIDIA GPU, install the rest
#    .\install.ps1 -Cpu       CPU-only PyTorch (smoke test only)
#    .\install.ps1 -NoVenv    use the current Python environment
#    .\install.ps1 -Yes       do not ask before running the PyTorch install command
# =============================================================================================
param([switch]$Cpu, [switch]$NoVenv, [switch]$Yes)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = "python"
& $py -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.10 or newer is required (python.org or the Microsoft Store)." }

if (-not $NoVenv) {
    if (-not (Test-Path ".venv")) { Write-Host "==> Creating .venv"; & $py -m venv .venv }
    & .\.venv\Scripts\Activate.ps1
    Write-Host "==> Using .venv (activate later with .\.venv\Scripts\Activate.ps1)"
}
& $py -m pip install --upgrade pip | Out-Null

if ($Cpu) {
    Write-Warning "CPU-only PyTorch: fine for configs/smoke_test.yaml, far too slow for real training."
    $torchCmd = "$py -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
    $gpu = (& nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1).Trim()
    $major = [int]($driver.Split(".")[0])
    Write-Host "==> NVIDIA GPU: $gpu, driver $driver"
    $cuda = if ($major -ge 570) { "cu128" } elseif ($major -ge 560) { "cu126" } elseif ($major -ge 550) { "cu124" } else { "cu121" }
    if ($major -lt 525) { Write-Warning "Driver $driver is older than CUDA 12 needs; update the NVIDIA driver." }
    $torchCmd = "$py -m pip install torch --index-url https://download.pytorch.org/whl/$cuda"
} else {
    throw "No NVIDIA GPU found (nvidia-smi missing). Training needs a GPU; for a CPU smoke test use -Cpu. Manual install: https://pytorch.org/get-started/locally/"
}

& $py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() or $([int][bool]$Cpu) else 1)" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "==> PyTorch already installed and working; skipping."
} else {
    Write-Host "`n  PyTorch install command:`n      $torchCmd`n  (other builds: https://pytorch.org/get-started/locally/)`n"
    if (-not $Yes) { $a = Read-Host "Run it now? [Y/n]"; if ($a -and $a -notmatch "^[Yy]") { throw "Install PyTorch yourself, then re-run." } }
    Invoke-Expression $torchCmd
}

& $py -c "import torch; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); raise SystemExit(0 if v>=(2,4) else 1)"
if ($LASTEXITCODE -ne 0) { throw "PyTorch >= 2.4 is required." }

Write-Host "==> Installing the remaining requirements"
& $py -m pip install -r requirements.txt
& $py -m pip install -e . | Out-Null
& $py -c "import torch, latentgen; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available(), '| latentgen', latentgen.__version__)"
Write-Host "`n==> Done. Try:  python scripts\make_smoke_data.py ; .\scripts\run_pipeline.ps1 configs\smoke_test.yaml"
