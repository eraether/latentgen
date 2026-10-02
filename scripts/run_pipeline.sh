#!/usr/bin/env bash
# Run every stage in order with one config.
#
#   bash scripts/run_pipeline.sh configs/ffhq512.yaml
#   bash scripts/run_pipeline.sh configs/smoke_test.yaml
#   bash scripts/run_pipeline.sh configs/ffhq512.yaml --from 2      # start at stage 2 (encoded data exists)
#
# Stages: 1a vqvae, 1b autoencoder, 1c encode, 2 maskgit, 3 cgan, 4 generate.
# Stages 1a/1b and 2/3 are independent pairs; on one GPU they simply run one after the other.
set -euo pipefail

CONFIG="${1:-configs/ffhq512.yaml}"
shift || true
[[ "$CONFIG" = /* ]] || CONFIG="$(pwd)/$CONFIG"   # keep a relative path valid after the cd below
FROM="1a"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
order=(1a 1b 1c 2 3 4)
started=false
for stage in "${order[@]}"; do
  [[ "$stage" == "$FROM" ]] && started=true
  $started || continue
  echo "================================================================ stage $stage"
  case "$stage" in
    1a) "$PY" scripts/01a_train_vqvae.py       --config "$CONFIG" ;;
    1b) "$PY" scripts/01b_train_autoencoder.py --config "$CONFIG" ;;
    1c) "$PY" scripts/01c_encode_dataset.py    --config "$CONFIG" ;;
    2)  "$PY" scripts/02_train_maskgit.py      --config "$CONFIG" ;;
    3)  "$PY" scripts/03_train_cgan.py         --config "$CONFIG" ;;
    4)  "$PY" scripts/04_generate.py           --config "$CONFIG" ;;
  esac
done
echo "pipeline finished"
