#!/usr/bin/env bash
# ==============================================================================
# TokaBench-GS: Standalone Film Packager
# Packages the self-contained cinematic presentation into a clean archive
# without any symlinks, ready for offline presentation and submission.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DIST_DIR="${SCRIPT_DIR}/film_portable"
ARCHIVE_ZIP="${SCRIPT_DIR}/film_portable.zip"

echo "=== [1/4] Preparing portable bundle in: ${DIST_DIR} ==="
rm -rf "${DIST_DIR}" "${ARCHIVE_ZIP}"
mkdir -p "${DIST_DIR}"

echo "=== [2/4] Copying film core files ==="
cp "${SCRIPT_DIR}/index.html" "${DIST_DIR}/"
cp "${SCRIPT_DIR}/README.md" "${DIST_DIR}/"
cp "${SCRIPT_DIR}/sources.json" "${DIST_DIR}/"
cp "${SCRIPT_DIR}/STORYBOARD.md" "${DIST_DIR}/"

echo "=== [3/4] Copying data and media assets (dereferencing symlinks) ==="
cp -R -L "${SCRIPT_DIR}/data" "${DIST_DIR}/"
cp -R -L "${SCRIPT_DIR}/assets" "${DIST_DIR}/"

# Verify no symlinks exist in the portable bundle
SYMLINK_COUNT=$(find "${DIST_DIR}" -type l | wc -l | tr -d ' ')
if [ "${SYMLINK_COUNT}" -ne 0 ]; then
  echo "❌ ERROR: Found ${SYMLINK_COUNT} symlinks in portable bundle!"
  find "${DIST_DIR}" -type l
  exit 1
fi
echo "✅ Zero symlinks verified."

echo "=== [4/4] Creating ZIP archive ==="
(cd "${SCRIPT_DIR}" && zip -r -q "film_portable.zip" "film_portable")

BUNDLE_SIZE=$(du -sh "${DIST_DIR}" | awk '{print $1}')
ZIP_SIZE=$(du -sh "${ARCHIVE_ZIP}" | awk '{print $1}')

echo "=============================================================================="
echo "🎉 Film bundle packaged successfully!"
echo "   Portable directory: ${DIST_DIR} (${BUNDLE_SIZE})"
echo "   Portable ZIP:       ${ARCHIVE_ZIP} (${ZIP_SIZE})"
echo "   To launch offline:  open ${DIST_DIR}/index.html"
echo "=============================================================================="
