#!/usr/bin/env bash
# =============================================================================================
#  latentgen installer (Linux / WSL)
#
#    ./install.sh              create .venv, install PyTorch for your GPU, install everything else
#    ./install.sh --no-venv    use the current Python environment instead of creating .venv
#    ./install.sh --cpu        CPU-only PyTorch (only for configs/smoke_test.yaml; training needs a GPU)
#    ./install.sh --yes        do not ask before running the PyTorch install command
#
#  PyTorch is the one dependency that must match your hardware, so this script detects your GPU
#  driver, prints the matching `pip install torch ...` command, and runs it after you confirm.
#  Everything else comes from requirements.txt.
# =============================================================================================
set -euo pipefail
cd "$(dirname "$0")"

USE_VENV=1; CPU=0; YES=0
for arg in "$@"; do
  case "$arg" in
    --no-venv) USE_VENV=0 ;;
    --cpu) CPU=1 ;;
    --yes|-y) YES=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 1 ;;
  esac
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*"; exit 1; }

# ------------------------------------------------------------------------------- python
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || die "python3 not found. Install Python 3.10+ first."
"$PY" - <<'EOF' || die "Python 3.10 or newer is required."
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF
say "Python: $("$PY" --version) ($(command -v "$PY"))"

if [[ $USE_VENV -eq 1 ]]; then
  if [[ ! -d .venv ]]; then
    say "Creating virtual environment .venv"
    "$PY" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=python
  say "Using virtual environment .venv (activate later with: source .venv/bin/activate)"
fi
"$PY" -m pip install --upgrade pip >/dev/null

# ------------------------------------------------------------------------------- GPU detection
TORCH_CMD=""
if [[ $CPU -eq 1 ]]; then
  warn "CPU-only install requested: fine for the smoke test, far too slow for real training."
  TORCH_CMD="$PY -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
elif command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
  MAJOR=${DRIVER%%.*}
  say "NVIDIA GPU: ${GPU:-unknown}, driver ${DRIVER:-unknown}"
  # Each CUDA wheel needs a minimum driver. Newer drivers run older CUDA builds, so pick the newest that fits.
  if   [[ -n "$MAJOR" && "$MAJOR" -ge 570 ]]; then CUDA=cu128
  elif [[ -n "$MAJOR" && "$MAJOR" -ge 560 ]]; then CUDA=cu126
  elif [[ -n "$MAJOR" && "$MAJOR" -ge 550 ]]; then CUDA=cu124
  elif [[ -n "$MAJOR" && "$MAJOR" -ge 525 ]]; then CUDA=cu121
  else
    warn "Driver $DRIVER is older than CUDA 12 needs. Update the NVIDIA driver, then re-run."
    CUDA=cu121
  fi
  TORCH_CMD="$PY -m pip install torch --index-url https://download.pytorch.org/whl/$CUDA"
elif command -v rocm-smi >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then
  say "AMD ROCm detected"
  TORCH_CMD="$PY -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.3"
  warn "FlashAttention is NVIDIA-only; the code falls back to standard attention (slower) on ROCm."
else
  die "No NVIDIA (nvidia-smi) or AMD (rocm-smi) GPU found. This pipeline needs a GPU to train.
       - On WSL: install the Windows NVIDIA driver; nvidia-smi then works inside WSL.
       - For a CPU smoke test only: ./install.sh --cpu
       - Manual PyTorch install for other setups: https://pytorch.org/get-started/locally/"
fi

# ------------------------------------------------------------------------------- PyTorch
if "$PY" -c "import torch, sys; sys.exit(0 if (torch.cuda.is_available() or $CPU) else 1)" 2>/dev/null; then
  say "PyTorch $("$PY" -c 'import torch; print(torch.__version__)') already installed and working; skipping."
else
  echo
  echo "  PyTorch install command:"
  echo "      $TORCH_CMD"
  echo "  (if this fails or you need a different build, use the selector at https://pytorch.org/get-started/locally/)"
  echo
  if [[ $YES -eq 0 ]]; then
    read -r -p "Run it now? [Y/n] " answer
    [[ -z "$answer" || "$answer" =~ ^[Yy] ]] || die "Install PyTorch yourself, then run ./install.sh again."
  fi
  $TORCH_CMD
fi

"$PY" - <<'EOF' || die "PyTorch >= 2.4 is required (nn.RMSNorm, torch.compile)."
import torch
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
raise SystemExit(0 if (major, minor) >= (2, 4) else 1)
EOF

# ------------------------------------------------------------------------------- everything else
say "Installing the remaining requirements"
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e . >/dev/null
say "Installed latentgen in editable mode"

# ------------------------------------------------------------------------------- verify
"$PY" - <<'EOF'
import torch
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 1024**3:.1f} GB VRAM, bf16 supported: {torch.cuda.is_bf16_supported()}")
    if p.total_memory / 1024**3 < 6:
        print("NOTE: under 6 GB VRAM -- lower microbatch_size and turn on activation_checkpointing (see README).")
import latentgen, schedulefree, yaml, tqdm, PIL  # noqa: F401
print("latentgen", latentgen.__version__, "imports fine")
EOF

echo
say "Done. Next steps:"
echo "    python scripts/make_smoke_data.py && bash scripts/run_pipeline.sh configs/smoke_test.yaml   # 2-minute end-to-end check"
echo "    edit configs/my_dataset.yaml, then: bash scripts/run_pipeline.sh configs/my_dataset.yaml"
