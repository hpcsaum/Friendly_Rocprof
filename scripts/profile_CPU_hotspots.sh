#!/usr/bin/env bash
# Launch a lightweight rocprof-sys CPU-sampling profile of a single command, then
# (by default) generate a short hotspots.txt via postprocess/extract_CPU_hotspots.py.
#
# rocprof-sys-sample wraps exactly one process. For MPI runs, put mpirun/srun
# *before* this script so each rank independently wraps its own process, e.g.:
#
#   mpirun -np 4 scripts/profile_CPU_hotspots.sh -o results/run1 -- ./app arg1 arg2
#
# All ranks share the same -o output directory; rocprof-sys's own default
# per-PID file naming keeps their output from colliding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/../postprocess/extract_CPU_hotspots.py"

OUTPUT_DIR="rocprof-sys-hotspots-output"
FREQ_HZ=100
RUN_SUMMARY=1
DRY_RUN=0
SELECTION_ARGS=()

usage() {
  cat <<'EOF'
Usage: profile_CPU_hotspots.sh [options] -- <command> [args...]

Runs your program once and measures which of its functions spend the most
time on the CPU -- including time the CPU spends just waiting for the GPU
to finish something. The result is a short, ranked text report telling you
where the time actually goes, so you know what's worth investigating first.

This is useful for finding CPU-side work that might be worth moving to the
GPU ("offloading"), or CPU code that's simply slow. It does NOT tell you
which GPU kernels are slow on the GPU itself -- for that, use
profile_GPU_hotspots.sh, or profile_hotspots.sh for both at once.

Numbers are reported as percentages of total measured time, good enough to
spot your top bottleneck -- not a precise, reproducible benchmark. Safe to
run repeatedly; it only observes your program, it doesn't change it.

Options:
  -o, --output-dir DIR   rocprof-sys output directory (default: rocprof-sys-hotspots-output)
  -f, --freq HZ           sampling frequency in Hz (default: 100)
  --top N                 hotspots per section to report (default: 20; last of --top/--threshold/--all wins)
  --threshold PCT         only report entries at or above PCT% of total runtime
  --all                   report every entry, no truncation
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
      python3 "$EXTRACTOR" "$OUTPUT_DIR" "${SELECTION_ARGS[@]}" || \
        echo "warning: hotspots extractor failed; profiling data is still in $OUTPUT_DIR" >&2
    else
      echo "warning: python3 not found, skipping hotspots summary; run it manually later:" >&2
      echo "  python3 $EXTRACTOR $OUTPUT_DIR ${SELECTION_ARGS[*]}" >&2
    fi
  fi
fi

exit "$APP_EXIT"
