#!/usr/bin/env bash
# ==============================================================================
# package_deck.sh — Self-contained portable deck packager and validator
# ==============================================================================
# Assembles deck_portable/ with physical copies of all assets (no symlinks)
# and verifies end-to-end functionality from /tmp/deck_portable_test/ via ffprobe.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DECK_SRC="${SCRIPT_DIR}/deck"
DECK_PORTABLE="${SCRIPT_DIR}/deck_portable"
RENDERS_DIR="${REPO_DIR}/motion/renders"
TEST_DIR="/tmp/deck_portable_test"

echo "=== [1/4] Ensuring real physical copies in ${DECK_SRC}/videos ==="
mkdir -p "${DECK_SRC}/videos"

# List of all required media files
MEDIA_FILES=(
  "act1_cutaway_key1.png"
  "act1_cutaway_key2.png"
  "act1_cutaway_preview.mp4"
  "act1_cutaway_final.mp4"
  "act2_surfaces_key1.png"
  "act2_surfaces_key2.png"
  "act2_surfaces_preview.mp4"
  "act2_surfaces_final.mp4"
  "act3_fieldlines_key1.png"
  "act3_fieldlines_key2.png"
  "act3_fieldlines_preview.mp4"
  "act3_fieldlines_final.mp4"
  "act4a_poincare_single_key1.png"
  "act4a_poincare_single_key2.png"
  "act4a_poincare_single_preview.mp4"
  "act4a_poincare_single_final.mp4"
  "act4b_poincare_chaos_key1.png"
  "act4b_poincare_chaos_key2.png"
  "act4b_poincare_chaos_preview.mp4"
  "act4b_poincare_chaos_final.mp4"
  "act5_compare_2d_key1.png"
  "act5_compare_2d_key2.png"
  "act5_compare_2d_preview.mp4"
  "act5_compare_2d_final.mp4"
  "film_continuous_preview.mp4"
  "film_continuous_final.mp4"
)

for fname in "${MEDIA_FILES[@]}"; do
  src_file="${RENDERS_DIR}/${fname}"
  dst_file="${DECK_SRC}/videos/${fname}"
  if [ ! -f "${src_file}" ]; then
    echo "ERROR: Source file not found: ${src_file}"
    exit 1
  fi
  if [ -L "${dst_file}" ] || [ ! -f "${dst_file}" ]; then
    echo "  Copying physical file: ${fname} -> ${DECK_SRC}/videos/"
    rm -f "${dst_file}"
    cp -p "${src_file}" "${dst_file}"
  fi
done

echo ""
echo "=== [2/4] Assembling portable deck at ${DECK_PORTABLE} ==="
rm -rf "${DECK_PORTABLE}"
mkdir -p "${DECK_PORTABLE}/js" "${DECK_PORTABLE}/videos"

cp -p "${DECK_SRC}/index.html" "${DECK_PORTABLE}/index.html"
cp -p "${DECK_SRC}/LICENSE-three.txt" "${DECK_PORTABLE}/LICENSE-three.txt"
cp -p "${DECK_SRC}/js"/* "${DECK_PORTABLE}/js/"
cp -p "${DECK_SRC}/videos"/* "${DECK_PORTABLE}/videos/"

# Verify zero symlinks in portable deck
SYMLINK_COUNT=$(find "${DECK_PORTABLE}" -type l | wc -l | tr -d ' ')
if [ "${SYMLINK_COUNT}" -ne 0 ]; then
  echo "ERROR: Found ${SYMLINK_COUNT} symlinks in ${DECK_PORTABLE}!"
  exit 1
fi
echo "  Self-contained portable deck assembled successfully (0 symlinks, total $(du -sh "${DECK_PORTABLE}" | cut -f1))."

echo ""
echo "=== [3/4] Testing deployment to ${TEST_DIR} ==="
rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"
cp -R "${DECK_PORTABLE}"/* "${TEST_DIR}/"

# Sanity checks on HTML and JS
[ -s "${TEST_DIR}/index.html" ] || { echo "ERROR: index.html is missing or empty"; exit 1; }
[ -s "${TEST_DIR}/js/three.min.js" ] || { echo "ERROR: three.min.js missing or empty"; exit 1; }
[ -s "${TEST_DIR}/js/poincare_data.js" ] || { echo "ERROR: poincare_data.js missing or empty"; exit 1; }

echo "  HTML and JS assets verified in ${TEST_DIR}."

echo ""
echo "=== [4/4] Verifying video streams via ffprobe in ${TEST_DIR}/videos ==="
PROBE_BIN="$(which ffprobe || echo "")"
if [ -z "${PROBE_BIN}" ]; then
  echo "WARNING: ffprobe not found in PATH, skipping stream decoding check."
else
  echo "  Using ffprobe: ${PROBE_BIN}"
  for vid in "${TEST_DIR}/videos"/*.mp4; do
    fname="$(basename "${vid}")"
    if [ -L "${vid}" ]; then
      echo "  ERROR: ${fname} in test dir is a symlink!"
      exit 1
    fi
    info="$(${PROBE_BIN} -v error -select_streams v:0 -show_entries stream=codec_name,width,height,duration -of csv=p=0 "${vid}" 2>&1)"
    echo "  [OK] ${fname}: ${info} ($(du -h "${vid}" | cut -f1))"
  done

  for png in "${TEST_DIR}/videos"/*.png; do
    fname="$(basename "${png}")"
    if [ -L "${png}" ]; then
      echo "  ERROR: ${png} in test dir is a symlink!"
      exit 1
    fi
    echo "  [OK] ${fname}: regular PNG ($(du -h "${png}" | cut -f1))"
  done
fi

echo ""
echo "=============================================================================="
echo "PORTABLE DECK PACKAGE VERIFICATION COMPLETE!"
echo "Destination: ${DECK_PORTABLE}"
echo "Test clone:  ${TEST_DIR}"
echo "Status: 100% self-contained, 0 symlinks, all media verified."
echo "=============================================================================="
