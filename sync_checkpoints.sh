#!/usr/bin/env bash
# Usage: ./sync_checkpoints.sh <modal-volume> <run-name> <file-prefix>
# Example: ./sync_checkpoints.sh kernelrl-runs qwen3-8b-kbl1-multiturn-v1-0p45t-3em5lr kbl1-multiturn-v1

set -euo pipefail

MODAL_VOLUME="$1"
RUN_NAME="$2"
FILE_PREFIX="$3"

if [[ -z "$MODAL_VOLUME" || -z "$RUN_NAME" || -z "$FILE_PREFIX" ]]; then
    echo "Usage: $0 <modal-volume> <run-name> <file-prefix>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINTS_DIR="$SCRIPT_DIR/checkpoints"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "==> Downloading $RUN_NAME from modal volume '$MODAL_VOLUME'..."
modal volume get "$MODAL_VOLUME" "$RUN_NAME" "$TMP_DIR/"

# modal volume get may place files under TMP_DIR/RUN_NAME or directly under TMP_DIR
if [[ -d "$TMP_DIR/$RUN_NAME" ]]; then
    SRC_DIR="$TMP_DIR/$RUN_NAME"
else
    SRC_DIR="$TMP_DIR"
fi

mkdir -p "$CHECKPOINTS_DIR"

# Find and install step_N directories
new_checkpoints=()
for step_dir in "$SRC_DIR"/step_*/; do
    [[ -d "$step_dir" ]] || continue
    step_name=$(basename "$step_dir")
    step_num="${step_name#step_}"
    dest="$CHECKPOINTS_DIR/${FILE_PREFIX}-s${step_num}"
    if [[ -d "$dest" ]]; then
        echo "  [skip] $dest already exists"
    else
        echo "  [install] $step_name -> $dest"
        mv "$step_dir" "$dest"
        new_checkpoints+=("$dest")
    fi
done

if [[ ${#new_checkpoints[@]} -eq 0 ]]; then
    echo "==> No new checkpoints to analyze."
    exit 0
fi

echo ""
echo "==> Running analyze_checkpoint.py on ${#new_checkpoints[@]} new checkpoint(s)..."
cd "$SCRIPT_DIR"
for ckpt in "${new_checkpoints[@]}"; do
    echo ""
    echo "---- $(basename "$ckpt") ----"
    python analyze_checkpoint.py --checkpoint "$ckpt"
done
