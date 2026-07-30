#!/usr/bin/env bash
# Launch a lightweight rocprof-sys CPU-sampling profile of a single command, then
# (by default) generate a short hotspots.txt via postprocess/rocprof_sys_hotspots.py.
#
# rocprof-sys-sample wraps exactly one process. For MPI runs, put mpirun/srun
# *before* this script so each rank independently wraps its own process, e.g.:
#
#   mpirun -np 4 scripts/rocprof_sys_profile.sh -o results/run1 -- ./app arg1 arg2
#
# All ranks share the same -o output directory; rocprof-sys's own default
# per-PID file naming keeps their output from colliding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/../postprocess/rocprof_sys_hotspots.py"

OUTPUT_DIR="rocprof-sys-hotspots-output"
FREQ_HZ=100
TOP_N=20
RUN_SUMMARY=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: rocprof_sys_profile.sh [options] -- <command> [args...]

Options:
  -o, --output-dir DIR   rocprof-sys output directory (default: rocprof-sys-hotspots-output)
  -f, --freq HZ           sampling frequency in Hz (default: 100)
  --top N                 hotspots per section to report (default: 20)
  --no-summary            skip auto-running the hotspots extractor afterwards
  --dry-run               print the command and env vars that would run, don't execute
  -h, --help              show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    -f|--freq)
      FREQ_HZ="$2"; shift 2 ;;
    --top)
      TOP_N="$2"; shift 2 ;;
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

if ! command -v rocprof-sys-sample >/dev/null 2>&1; then
  echo "error: 'rocprof-sys-sample' not found on PATH." >&2
  echo "       Load the ROCm module providing rocprofiler-systems (e.g. 'module load rocm') and retry." >&2
  exit 1
fi

export ROCPROFSYS_OUTPUT_PATH="$OUTPUT_DIR"
export ROCPROFSYS_TEXT_OUTPUT=1
export ROCPROFSYS_JSON_OUTPUT=1
export ROCPROFSYS_FLAT_PROFILE=1
export ROCPROFSYS_TRACE=0

CMD=(rocprof-sys-sample -f "$FREQ_HZ" -- "$@")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would export:"
  echo "  ROCPROFSYS_OUTPUT_PATH=$ROCPROFSYS_OUTPUT_PATH"
  echo "  ROCPROFSYS_TEXT_OUTPUT=$ROCPROFSYS_TEXT_OUTPUT"
  echo "  ROCPROFSYS_JSON_OUTPUT=$ROCPROFSYS_JSON_OUTPUT"
  echo "  ROCPROFSYS_FLAT_PROFILE=$ROCPROFSYS_FLAT_PROFILE"
  echo "  ROCPROFSYS_TRACE=$ROCPROFSYS_TRACE"
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
      python3 "$EXTRACTOR" "$OUTPUT_DIR" -n "$TOP_N" || \
        echo "warning: hotspots extractor failed; profiling data is still in $OUTPUT_DIR" >&2
    else
      echo "warning: python3 not found, skipping hotspots summary; run it manually later:" >&2
      echo "  python3 $EXTRACTOR $OUTPUT_DIR -n $TOP_N" >&2
    fi
  fi
fi

exit "$APP_EXIT"
