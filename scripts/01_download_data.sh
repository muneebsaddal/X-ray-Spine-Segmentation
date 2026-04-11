#!/usr/bin/env bash
# scripts/01_download_data.sh
#
# Downloads a small subset of the VerSe2020 public dataset for the POC.
# VerSe2020: https://github.com/anjany/verse
#
# For the full project, replace SAMPLE_IDS with the complete dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$PROJECT_ROOT/data/raw/verse_sample"

mkdir -p "$RAW_DIR"

echo "============================================"
echo "  TotalSpineSeg 2D X-Ray — Data Download"
echo "============================================"
echo "Target: $RAW_DIR"
echo ""

# VerSe2020 training data is hosted on Zenodo (DOI: 10.5281/zenodo.3752676)
# We download a small representative subset for the POC (3 cases).
#
# Full dataset DOI: https://doi.org/10.5281/zenodo.3752676

ZENODO_BASE="https://zenodo.org/record/3752676/files"

# These are publicly available VerSe2020 sample files
# (sub-verse004, sub-verse009, sub-verse020 — lateral-friendly FOVs)
declare -a SAMPLE_IDS=("sub-verse004" "sub-verse009" "sub-verse020")

DOWNLOAD_SUCCESS=0

# Try downloading from VerSe GitHub releases (smaller test files)
VERSE_GITHUB="https://github.com/anjany/verse/releases/download/v1.0"

echo "[INFO] Attempting to download VerSe sample data..."
echo "[INFO] If download fails, synthetic data will be generated automatically."
echo ""

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
    IMAGE_FILE="${SAMPLE_ID}_dir-sag_ct.nii.gz"
    LABEL_FILE="${SAMPLE_ID}_dir-sag_seg-vert_msk.nii.gz"

    # Try multiple mirrors
    for BASE_URL in \
        "https://raw.githubusercontent.com/anjany/verse/main/data" \
        "$ZENODO_BASE"; do

        if curl -sfL --max-time 30 \
            "${BASE_URL}/${IMAGE_FILE}" \
            -o "${RAW_DIR}/${IMAGE_FILE}" 2>/dev/null; then
            echo "[OK] Downloaded image: $IMAGE_FILE"
            DOWNLOAD_SUCCESS=$((DOWNLOAD_SUCCESS + 1))
            break
        fi
    done
done

echo ""

if [ "$DOWNLOAD_SUCCESS" -eq 0 ]; then
    echo "[INFO] Could not reach VerSe servers (expected in offline/restricted environments)."
    echo "[INFO] Generating synthetic phantom data for POC demonstration..."
    echo ""
    python3 "$PROJECT_ROOT/src/generate_phantom.py" \
        --output "$RAW_DIR" \
        --num_cases 3
    echo "[OK] Synthetic phantom data created at: $RAW_DIR"
else
    echo "[OK] Downloaded $DOWNLOAD_SUCCESS file(s) to $RAW_DIR"
fi

echo ""
echo "Next step:"
echo "  python src/generate_drr.py --input data/raw/verse_sample --output data/processed/xray_synthetic"
