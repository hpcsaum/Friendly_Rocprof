#!/usr/bin/env bash
# Wraps the given command in a debugger from the moment it launches, so a fatal
# signal (segfault, abort, ...) is caught live and a full backtrace is printed
# before the process is gone -- rather than the process just disappearing.
#
# For MPI runs, this script is still called exactly once -- pass the MPI launch
# command as data via --mpi "<launch cmd>" (e.g. --mpi "mpirun -np 4"), and this
# script places it in front of the debugged run for you, e.g.:
#
#   scripts/debug_crash.sh --mpi "mpirun -np 4" -o results/run1 -- ./app arg1 arg2
#
# Each rank needs to resolve its own identity and write its own backtrace file
# at its own runtime (not this script's), so the MPI case wraps the debugger
# invocation in a small `bash -c` script executed once per rank, resolving rank
# via OMPI_COMM_WORLD_RANK / PMI_RANK / PMIX_RANK / SLURM_PROCID (falling back
# to PID if none of those are set).

set -euo pipefail

OUTPUT_DIR="debug_crash-output-$(date +%F_%H.%M.%S)"
DRY_RUN=0
MPI_STR=""

usage() {
  cat <<'EOF'
Usage: debug_crash.sh [options] -- <command> [args...]

Runs your program under a debugger from the moment it starts, so that if it
crashes -- segfault, abort, any fatal signal -- the debugger is already
attached and prints exactly where every thread was when it happened. Use
this once you already have a crash you can reproduce; it does NOT help with
a job that just hangs with no crash and no signal -- see debug_hang.py for
that case instead.

Every rank runs noticeably slower than a normal launch, because it's
wrapped in a debugger the whole time -- this is a one-off debugging run,
not something to leave in your normal job scripts. Works for a single
process, or, with --mpi, one debugger per MPI rank; each rank writes its
own backtrace file so nothing gets overwritten.

Under the hood, this uses AMD's rocgdb -- see
https://rocm.docs.amd.com/projects/ROCgdb/en/latest/ for details.

Options:
  -o, --output-dir DIR   where backtrace file(s) go (default: debug_crash-output-<timestamp>)
                          non-MPI: DIR/backtrace.txt -- MPI: DIR/rank_<id>.bt per rank
  --mpi "<launch cmd>"    MPI launch command to prefix the run with (e.g. "mpirun -np 4");
                          omit to run a single rank with no MPI
  --dry-run               print the command that would run, don't execute
  -h, --help              show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
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

if ! command -v rocgdb >/dev/null 2>&1; then
  echo "error: 'rocgdb' not found on PATH." >&2
  echo "       Load the ROCm module providing rocgdb (e.g. 'module load rocm') and retry." >&2
  exit 1
fi

MPI_ARR=()
if [[ -n "$MPI_STR" ]]; then
  read -ra MPI_ARR <<< "$MPI_STR"
fi

GDB_ARGS=(-q --batch -nx -ex "set pagination off" -ex run -ex "thread apply all bt" -ex quit)

if [[ -n "$MPI_STR" ]]; then
  # Evaluated once per rank, by that rank's own shell -- $OUTPUT_DIR is spliced
  # in as a literal path at script-construction time, everything else resolves
  # at each rank's own runtime.
  RANK_SCRIPT='rank="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${PMIX_RANK:-${SLURM_PROCID:-$$}}}}"; exec rocgdb -q --batch -nx -ex "set pagination off" -ex run -ex "thread apply all bt" -ex quit --args "$@" > "'"$OUTPUT_DIR"'/rank_${rank}.bt" 2>&1'
  CMD=("${MPI_ARR[@]}" bash -c "$RANK_SCRIPT" _ "$@")
else
  CMD=(rocgdb "${GDB_ARGS[@]}" --args "$@")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would run:"
  printf '  %q ' "${CMD[@]}"
  if [[ -n "$MPI_STR" ]]; then
    echo
    echo "  (each rank writes its own $OUTPUT_DIR/rank_<id>.bt)"
  else
    echo "> $OUTPUT_DIR/backtrace.txt 2>&1"
  fi
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

set +e
if [[ -n "$MPI_STR" ]]; then
  "${CMD[@]}"
else
  "${CMD[@]}" > "$OUTPUT_DIR/backtrace.txt" 2>&1
fi
GDB_EXIT=$?
set -e

exit "$GDB_EXIT"
