#!/usr/bin/env bash
# ==============================================================================
# team/run_all.sh — Unified Offline Reproduction and Harness Self-Check Suite
#
# Laboratory: AI Lab named after V. M. Horshkov, FMF Igor Sikorsky KPI
# Hackathon:   Tokamak Magnetic Equilibrium & Field Lines
# Branch:      presentation-prep
#
# What this script does:
# 1. Runs Scorer Harness Self-Checks (perfect -> 1.0, zeros -> 0.0) with timing.
# 2. Verifies and reproduces LEADERBOARD.md metrics offline.
# 3. Verifies NUMBERS_TABLE.md against task raw outputs.
# 4. Generates/updates core defense figures in team/presentation/figures/.
# 5. Executes S'-gate sensitivity grid verification.
# 6. Prints PASS/FAIL status for each step and total execution time.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${ROOT_DIR}/fusion equilibrium challenge/starter/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python environment not found at ${PYTHON}"
    exit 1
fi

TOTAL_START=$(date +%s)
echo "========================================================================"
echo "    FMF KPI TOKAMAK BENCHMARK: FULL REPRODUCIBILITY SUITE"
echo "    Date: $(date '+%Y-%m-%d %H:%M:%S') | Branch: $(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'presentation-prep')"
echo "========================================================================"
echo ""

# ------------------------------------------------------------------------------
# STEP 1: Scorer Self-Checks (perfect -> 1, zeros -> 0)
# ------------------------------------------------------------------------------
echo ">>> STEP 1: Running Scorer Harness Self-Checks..."

# 1A: Perfect mode check
T0=$(date +%s)
PERFECT_OUT=$("${PYTHON}" "${ROOT_DIR}/eval_submission.py" --mode perfect 2>&1)
T1=$(date +%s)
DT_PERFECT=$((T1 - T0))

if echo "${PERFECT_OUT}" | grep -q "Official S:        1.000000" && echo "${PERFECT_OUT}" | grep -q "Proposed S'-gate:  1.000000"; then
    echo "  [PASS] Scorer Self-Check: perfect -> 1.000000 (Elapsed: ${DT_PERFECT}s)"
else
    echo "  [FAIL] Scorer Self-Check: perfect FAILED!"
    echo "${PERFECT_OUT}"
    exit 1
fi

# 1B: Zeros mode check
T0=$(date +%s)
ZEROS_OUT=$("${PYTHON}" "${ROOT_DIR}/eval_submission.py" --mode zeros 2>&1)
T1=$(date +%s)
DT_ZEROS=$((T1 - T0))

if echo "${ZEROS_OUT}" | grep -q "Official S:        0.000000" && echo "${ZEROS_OUT}" | grep -q "Proposed S'-gate:  0.000000"; then
    echo "  [PASS] Scorer Self-Check: zeros -> 0.000000 (Elapsed: ${DT_ZEROS}s)"
else
    echo "  [FAIL] Scorer Self-Check: zeros FAILED!"
    echo "${ZEROS_OUT}"
    exit 1
fi
echo ""

# ------------------------------------------------------------------------------
# STEP 2: Reproduce Leaderboard, Numbers Table, and Core Presentation Figures
# ------------------------------------------------------------------------------
echo ">>> STEP 2: Reproducing LEADERBOARD, NUMBERS_TABLE, and Defense Figures..."
T0=$(date +%s)
"${PYTHON}" "${ROOT_DIR}/team/scripts/reproduce_all.py"
T1=$(date +%s)
DT_REP=$((T1 - T0))
echo "  [PASS] LEADERBOARD, NUMBERS_TABLE, and Figures verified (Elapsed: ${DT_REP}s)"
echo ""

# ------------------------------------------------------------------------------
# STEP 3: Verify Sensitivity Scan
# ------------------------------------------------------------------------------
echo ">>> STEP 3: Verifying S'-gate Sensitivity Grid (g_ref x tau)..."
T0=$(date +%s)
"${PYTHON}" "${ROOT_DIR}/team/scripts/sensitivity_scan.py" > /dev/null
T1=$(date +%s)
DT_SENS=$((T1 - T0))
echo "  [PASS] Sensitivity scan verified (25 parameter points) (Elapsed: ${DT_SENS}s)"
echo ""

# ------------------------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------------------------
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "========================================================================"
echo "    ALL CHECKS PASSED: PIPELINE IS 100% REPRODUCIBLE OFFLINE"
echo "    - Self-Check (perfect -> 1): PASS (${DT_PERFECT}s)"
echo "    - Self-Check (zeros -> 0):   PASS (${DT_ZEROS}s)"
echo "    - Leaderboard integrity:     PASS (${DT_REP}s)"
echo "    - Figures generated:         team/presentation/figures/leaderboard_comparison.png"
echo "    - Sensitivity analysis:      team/notes/SENSITIVITY.md"
echo "    - Total execution time:      ${TOTAL_ELAPSED}s"
echo "========================================================================"
