#!/usr/bin/env bash
# Launch a rocprofv3 GPU-kernel profile of a single command, then (by default)
# generate a short hotspots.txt via postprocess/tools/extract_GPU_hotspots.py.
#
# rocprofv3 wraps exactly one process, transparently (no rebuild/instrumentation
# needed). For MPI runs, this script is still called exactly once -- pass the MPI
# launch command as data via --mpi "<launch cmd>" (e.g. --mpi "mpirun -np 4"), and
# this script places it in front of the profiling run for you, e.g.:
#
#   scripts/profile_GPU_hotspots.sh --mpi "mpirun -np 4" -o results/run1 -- ./app arg1 arg2
#
# All ranks share the same -o output directory; rocprofv3's own default
# per-PID file naming keeps their output from colliding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/../postprocess/tools/extract_GPU_hotspots.py"

OUTPUT_DIR="profile_GPU_hotspots-output-$(date +%F_%H.%M.%S)"
RUN_SUMMARY=1
DRY_RUN=0
MPI_STR=""
SELECTION_ARGS=()

usage() {
  cat <<'EOF'
Usage: profile_GPU_hotspots.sh [options] -- <command> [args...]

Runs your program once and measures which GPU kernels actually spend the
most time executing on the GPU itself. This is real device execution time,
not a CPU-side estimate -- it tells you which pieces of GPU work are worth
optimizing first (e.g. making faster, or launching less often).

It does NOT show CPU-side hotspots (functions still running on the CPU,
which might be candidates for offloading to the GPU in the first place) --
for that, use profile_CPU_hotspots.sh, or profile_hotspots.sh for both at
once.

Numbers are reported as percentages of total measured GPU time, good
enough to spot your top bottleneck -- not a precise, reproducible
benchmark. Safe to run repeatedly; it only observes your program, it
doesn't change it or require rebuilding it.

For MPI runs, the report also includes a kernel load-imbalance table
(each kernel's average/min/max time and how much it varies across ranks)
after the main hotspots table. If your program needs MPI to run at all,
pass --mpi "<launch command>" (e.g. --mpi "mpirun -np 4") -- this script is
still called exactly once; it places the launch command in front of the
profiling run for you.

Under the hood, this uses AMD's rocprofv3 -- see
https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html
for details.

Options:
  -o, --output-dir DIR   rocprofv3 output directory (default: profile_GPU_hotspots-output-<timestamp>)
  --top N                 hotspots to report (default: 20; last of --top/--threshold/--all wins)
  --threshold PCT         only report kernels at or above PCT% of total GPU time
                          (or, in the load-imbalance table, at or above PCT% coefficient of variation)
  --all                   report every kernel, no truncation
  --no-summary            skip auto-running the hotspots extractor afterwards
  --mpi "<launch cmd>"    MPI launch command to prefix the profiling run with (e.g. "mpirun -np 4")
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

if ! command -v rocprofv3 >/dev/null 2>&1; then
  echo "error: 'rocprofv3' not found on PATH." >&2
  echo "       Load the ROCm module providing rocprofiler-sdk (e.g. 'module load rocm') and retry." >&2
  exit 1
fi

MPI_ARR=()
if [[ -n "$MPI_STR" ]]; then
  read -ra MPI_ARR <<< "$MPI_STR"
fi

# --output-format csv is non-negotiable: rocprofv3's own default is rocpd
# (SQLite), which the extractor doesn't parse. --truncate-kernels keeps names
# readable, matching AMD's own recommended usage.
CMD=("${MPI_ARR[@]}" rocprofv3 --kernel-trace --stats --summary --truncate-kernels \
     --output-format csv -d "$OUTPUT_DIR" -- "$@")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would run:"
  printf '  %q ' "${CMD[@]}"
  echo
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

set +e
"${CMD[@]}"
APP_EXIT=$?
set -e

if [[ "$RUN_SUMMARY" -eq 1 ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 "$EXTRACTOR" "$OUTPUT_DIR" "${SELECTION_ARGS[@]}" || \
      echo "warning: hotspots extractor failed; profiling data is still in $OUTPUT_DIR" >&2
  else
    echo "warning: python3 not found, skipping hotspots summary; run it manually later:" >&2
    echo "  python3 $EXTRACTOR $OUTPUT_DIR ${SELECTION_ARGS[*]}" >&2
  fi
fi

exit "$APP_EXIT"
