#!/usr/bin/env bash
# Usage: ./sync_checkpoints.sh <modal-volume> <run-name> <local-dir> [--verbose]
# Example: ./sync_checkpoints.sh kernelrl-runs qwen3-8b-kbl1-mt-v2-0p6t-2em5lr kbl1-mt-v2
# Downloads step_N checkpoints from Modal into checkpoints/<local-dir>/step_N/

set -euo pipefail

MODAL_VOLUME="$1"
RUN_NAME="$2"
FILE_PREFIX="$3"
VERBOSE=0
for arg in "${@:4}"; do
    [[ "$arg" == "--verbose" ]] && VERBOSE=1
done

if [[ -z "$MODAL_VOLUME" || -z "$RUN_NAME" || -z "$FILE_PREFIX" ]]; then
    echo "Usage: $0 <modal-volume> <run-name> <file-prefix> [--verbose]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINTS_DIR="$SCRIPT_DIR/checkpoints/$FILE_PREFIX"
mkdir -p "$CHECKPOINTS_DIR"

# List steps available on Modal
echo "==> Listing steps in '$MODAL_VOLUME/$RUN_NAME'..."
available_steps=$(modal volume ls "$MODAL_VOLUME" "$RUN_NAME" 2>/dev/null | grep -oE 'step_[0-9]+' | sort -t_ -k2 -n)

if [[ -z "$available_steps" ]]; then
    echo "No step_N directories found on Modal volume."
    exit 0
fi

# Download only steps we don't already have locally
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

new_checkpoints=()
for step_name in $available_steps; do
    dest="$CHECKPOINTS_DIR/$step_name"
    if [[ -d "$dest" ]]; then
        echo "  [skip] $step_name already at $dest"
        continue
    fi
    echo "  [download] $step_name -> $dest"
    step_tmp="$TMP_DIR/$step_name"
    mkdir -p "$step_tmp"
    modal volume get "$MODAL_VOLUME" "$RUN_NAME/$step_name" "$step_tmp/"
    # modal volume get may nest under the basename
    if [[ -d "$step_tmp/$step_name" ]]; then
        mv "$step_tmp/$step_name" "$dest"
    else
        mv "$step_tmp" "$dest"
    fi
    new_checkpoints+=("$dest")
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
    if [[ "$VERBOSE" -eq 1 ]]; then
        python analyze_checkpoint.py --checkpoint "$ckpt"
    else
        python analyze_checkpoint.py --checkpoint "$ckpt" | grep -E "^(Loading|TOTAL|-)"
    fi
done
