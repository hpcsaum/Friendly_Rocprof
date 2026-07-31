#!/usr/bin/env bash
# Launch a rocprofv3 GPU-kernel profile of a single command, then (by default)
# generate a short hotspots.txt via postprocess/rocprofv3_hotspots.py.
#
# rocprofv3 wraps exactly one process, transparently (no rebuild/instrumentation
# needed). For MPI runs, put mpirun/srun *before* this script so each rank
# independently wraps its own process, e.g.:
#
#   mpirun -np 4 scripts/rocprofv3_profile.sh -o results/run1 -- ./app arg1 arg2
#
# All ranks share the same -o output directory; rocprofv3's own default
# per-PID file naming keeps their output from colliding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/../postprocess/rocprofv3_hotspots.py"

OUTPUT_DIR="rocprofv3-hotspots-output"
RUN_SUMMARY=1
DRY_RUN=0
SELECTION_ARGS=()

usage() {
  cat <<'EOF'
Usage: rocprofv3_profile.sh [options] -- <command> [args...]

Options:
  -o, --output-dir DIR   rocprofv3 output directory (default: rocprofv3-hotspots-output)
  --top N                 hotspots to report (default: 20; last of --top/--threshold/--all wins)
  --threshold PCT         only report kernels at or above PCT% of total GPU time
  --all                   report every kernel, no truncation
  --no-summary            skip auto-running the hotspots extractor afterwards
  --dry-run               print the command that would run, don't execute
  -h, --help              show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --top)
      SELECTION_ARGS=(--top "$2"); shift 2 ;;
    --threshold)
      SELECTION_ARGS=(--threshold "$2"); shift 2 ;;
    --all)
      SELECTION_ARGS=(--all); shift ;;
    --no-summary)
      RUN_SUMMARY=0; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; break ;;
    *)
      echo "error: unknown option '$1'" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "error: no command given after --" >&2
  usage >&2
  exit 1
fi

if ! command -v rocprofv3 >/dev/null 2>&1; then
  echo "error: 'rocprofv3' not found on PATH." >&2
  echo "       Load the ROCm module providing rocprofiler-sdk (e.g. 'module load rocm') and retry." >&2
  exit 1
fi

# --output-format csv is non-negotiable: rocprofv3's own default is rocpd
# (SQLite), which the extractor doesn't parse. --truncate-kernels keeps names
# readable, matching AMD's own recommended usage.
CMD=(rocprofv3 --kernel-trace --stats --summary --truncate-kernels \
     --output-format csv -d "$OUTPUT_DIR" -- "$@")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would run:"
  printf '  %q ' "${CMD[@]}"
  echo
  exit 0
fi

set +e
"${CMD[@]}"
APP_EXIT=$?
set -e

if [[ "$RUN_SUMMARY" -eq 1 ]]; then
  RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}"
  if [[ "$RANK" -eq 0 ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python3 "$EXTRACTOR" "$OUTPUT_DIR" "${SELECTION_ARGS[@]}" || \
        echo "warning: hotspots extractor failed; profiling data is still in $OUTPUT_DIR" >&2
    else
      echo "warning: python3 not found, skipping hotspots summary; run it manually later:" >&2
      echo "  python3 $EXTRACTOR $OUTPUT_DIR ${SELECTION_ARGS[*]}" >&2
    fi
  fi
fi

exit "$APP_EXIT"
