#!/usr/bin/env bash
# Launch a lightweight rocprof-sys CPU-sampling profile of a single command, then
# (by default) generate a short hotspots.txt via postprocess/extract_CPU_hotspots.py
# and a calltree.txt via postprocess/extract_calltree.py.
#
# rocprof-sys-sample wraps exactly one process. For MPI runs, this script is still
# called exactly once -- pass the MPI launch command as data via --mpi "<launch cmd>"
# (e.g. --mpi "mpirun -np 4"), and this script places it in front of the profiling
# run for you, e.g.:
#
#   scripts/profile_CPU_hotspots.sh --mpi "mpirun -np 4" -o results/run1 -- ./app arg1 arg2
#
# All ranks share the same -o output directory; rocprof-sys's own default
# per-PID file naming keeps their output from colliding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/../postprocess/extract_CPU_hotspots.py"
CALLTREE_EXTRACTOR="$SCRIPT_DIR/../postprocess/extract_calltree.py"

OUTPUT_DIR="profile_CPU_hotspots-output-$(date +%F_%H.%M.%S)"
FREQ_HZ=100
RUN_SUMMARY=1
DRY_RUN=0
MPI_STR=""
SELECTION_ARGS=()
UNFILTERED=0
CALLTREE_ARGS=()

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

Ranks functions by their own (self) time, not counting time spent in
functions they call -- so a function that just calls other functions won't
crowd out the ones that actually do the work. This needs real call-tree
data, which has somewhat more overhead than a flat profile; set
ROCPROFSYS_FLAT_PROFILE=1 yourself beforehand if you need the lighter-weight
(but self-time-blind) mode back for an overhead-sensitive run.

For MPI runs, the report also includes a load-imbalance table (each
function's average/min/max time and how much it varies across ranks,
including MPI calls) after the main hotspots tables. If your program needs
MPI to run at all, pass --mpi "<launch command>" (e.g. --mpi "mpirun -np 4")
-- this script is still called exactly once; it places the launch command in
front of the profiling run for you.

Under the hood, this uses AMD's rocprof-sys (ROCm Systems Profiler) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/ for details.

Options:
  -o, --output-dir DIR   rocprof-sys output directory (default: profile_CPU_hotspots-output-<timestamp>)
  -f, --freq HZ           sampling frequency in Hz (default: 100)
  --top N                 hotspots per section to report (default: 20; last of --top/--threshold/--all wins)
  --threshold PCT         only report entries at or above PCT% of total runtime
                          (or, in the load-imbalance table, at or above PCT% coefficient of variation)
  --all                   report every entry, no truncation
  --no-summary            skip auto-running the hotspots and calltree extractors afterwards
  --unfiltered            rank by inclusive (total) time instead of self time -- the old
                          behavior, where a function that just calls other functions can
                          still rank high
  --max-depth N           truncate the auto-generated call tree at this depth (default: unlimited)
  --show-gpu-api          include GPU-API/offload-runtime noise in the auto-generated call tree
                          (hidden by default, same as the hotspots report)
  --show-rocprofsys-internals  include rocprof-sys's own instrumentation/GOTCHA frames instead of
                          splicing them out of the auto-generated call tree
  --show-mpi-internals    include MPI library internals below the first MPI frame in the
                          auto-generated call tree instead of collapsing them
  --show-compiler-runtime include compiler-runtime allocator/intrinsic helper noise in the
                          auto-generated call tree
  --show-all-internals    shorthand for all four --show-* flags above at once
  --mpi "<launch cmd>"    MPI launch command to prefix the profiling run with (e.g. "mpirun -np 4")
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
    --unfiltered)
      UNFILTERED=1; shift ;;
    --max-depth)
      CALLTREE_ARGS+=(--max-depth "$2"); shift 2 ;;
    --show-gpu-api)
      CALLTREE_ARGS+=(--show-gpu-api); shift ;;
    --show-rocprofsys-internals)
      CALLTREE_ARGS+=(--show-rocprofsys-internals); shift ;;
    --show-mpi-internals)
      CALLTREE_ARGS+=(--show-mpi-internals); shift ;;
    --show-compiler-runtime)
      CALLTREE_ARGS+=(--show-compiler-runtime); shift ;;
    --show-all-internals)
      CALLTREE_ARGS+=(--show-all-internals); shift ;;
    --mpi)
      MPI_STR="$2"; shift 2 ;;
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
export ROCPROFSYS_FLAT_PROFILE=0
export ROCPROFSYS_TRACE=0

MPI_ARR=()
if [[ -n "$MPI_STR" ]]; then
  read -ra MPI_ARR <<< "$MPI_STR"
fi
UNFILTERED_ARGS=()
[[ "$UNFILTERED" -eq 1 ]] && UNFILTERED_ARGS=(--unfiltered)

CMD=("${MPI_ARR[@]}" rocprof-sys-sample -f "$FREQ_HZ" -- "$@")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would export:"
  echo "  ROCPROFSYS_OUTPUT_PATH=$ROCPROFSYS_OUTPUT_PATH"
  echo "  ROCPROFSYS_TEXT_OUTPUT=$ROCPROFSYS_TEXT_OUTPUT"
  echo "  ROCPROFSYS_JSON_OUTPUT=$ROCPROFSYS_JSON_OUTPUT"
  echo "  ROCPROFSYS_FLAT_PROFILE=$ROCPROFSYS_FLAT_PROFILE"
  echo "  ROCPROFSYS_TRACE=$ROCPROFSYS_TRACE"
  echo "  (hierarchical call-tree data -- needed for the extractor's self-time ranking; see -h)"
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
  if command -v python3 >/dev/null 2>&1; then
    python3 "$EXTRACTOR" "$OUTPUT_DIR" "${SELECTION_ARGS[@]}" "${UNFILTERED_ARGS[@]}" || \
      echo "warning: hotspots extractor failed; profiling data is still in $OUTPUT_DIR" >&2
    python3 "$CALLTREE_EXTRACTOR" "$OUTPUT_DIR" "${CALLTREE_ARGS[@]}" || \
      echo "warning: calltree extractor failed; profiling data is still in $OUTPUT_DIR" >&2
  else
    echo "warning: python3 not found, skipping hotspots/calltree summary; run manually later:" >&2
    echo "  python3 $EXTRACTOR $OUTPUT_DIR ${SELECTION_ARGS[*]} ${UNFILTERED_ARGS[*]}" >&2
    echo "  python3 $CALLTREE_EXTRACTOR $OUTPUT_DIR ${CALLTREE_ARGS[*]}" >&2
  fi
fi

exit "$APP_EXIT"
