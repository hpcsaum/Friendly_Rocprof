#!/usr/bin/env bash
# Finds the biggest GPU kernels (via scripts/profile_GPU_hotspots.sh / a saved hotspots.txt),
# then builds and runs a rocprof-compute command that profiles only those -- specifically each
# selected kernel's SECOND call (-d 2), skipping the first (unrepresentative first-touch/
# page-fault overhead), then auto-runs rocprof-compute analyze and surfaces its own output.
#
# A kernel dispatched only once has no second call to target and is excluded by default --
# pass --all-dispatches to include those too (see its own warning below; this can multiply
# profiling time significantly).
#
# MPI: rocprof-compute's own multi-rank output isolation (so concurrent ranks don't write into
# the same directory) is a real feature in some version of rocprof-compute, but is confirmed
# ABSENT through rocprofiler-compute 3.4.0 (ROCm 7.2.x) -- no released version this could be
# checked against has it. Rather than assume either way, this script probes the installed
# rocprof-compute's own `profile --help` output for evidence of it, and separately determines
# the actual rank count your --mpi launch command produces (by running it once against a
# trivial no-op) -- if more than one rank is detected and the installed rocprof-compute shows
# no sign of supporting that safely, this script refuses to run rather than risk every rank
# silently overwriting the same output directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_LAUNCHER="$SCRIPT_DIR/profile_GPU_hotspots.sh"
SELECTOR="$SCRIPT_DIR/../postprocess/tools/select_hotspot_kernels.py"

RUN_TS="$(date +%F_%H.%M.%S)"
OUTPUT_DIR="profile_hotspot_kernels-scan-$RUN_TS"
WORKLOAD_DIR="profile_hotspot_kernels-workload-$RUN_TS"
REPORT=""
SELECTION_ARGS=()
ALL_DISPATCHES=0
RUN_SUMMARY=1
MPI_STR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: profile_hotspot_kernels.sh [options] -- <command> [args...]

Runs your program, finds its biggest GPU kernels (reusing profile_GPU_hotspots.sh, or a report
you already have), and profiles ONLY those in hardware-counter-level detail -- specifically
each selected kernel's second call, skipping the first (its timing is usually thrown off by
one-time first-touch/page-fault overhead that doesn't represent steady-state cost). Kernels
that only run once are left out entirely by default: there's no second call to target.

This is the "go deep on what's already known to be hot" tool -- use profile_GPU_hotspots.sh
first if you don't already know your biggest kernels.

Options:
  -o, --output-dir DIR    directory for the auto-profiling scan (default: profile_hotspot_kernels-scan-<timestamp>)
  --report FILE             reuse an existing hotspots.txt instead of auto-profiling
  --top N                   hotspot kernels to select (default: 20; last of --top/--threshold/--all wins)
  --threshold PCT           only select kernels at or above PCT% of total GPU time
  --all                     select every kernel found, no truncation
  --workload-dir DIR        directory for rocprof-compute's own output (default: profile_hotspot_kernels-workload-<timestamp>)
  --all-dispatches          profile EVERY call of each selected kernel instead of just its 2nd --
                            for a kernel called N times this can multiply profiling time by
                            roughly N (on top of rocprof-compute's own multi-pass counter
                            collection). Only use this for a small test case specifically sized
                            for rocprof-compute; for a normal/long-running application, leave
                            this off and let the default (2nd call only) keep runtime bounded.
                            Also includes kernels that only ran once, which are excluded by default.
  --no-summary              skip auto-running rocprof-compute analyze afterwards
  --mpi "<launch cmd>"      MPI launch command, forwarded to the auto-profiling run and to
                            rocprof-compute itself -- see the MPI note below; this script checks
                            the installed rocprof-compute's own multi-rank support before using
                            this with more than one rank, and refuses to run if it can't confirm
                            it's safe.
  --dry-run                 print what would run, don't execute
  -h, --help                show this help

Under the hood, this uses AMD's rocprof-compute (ROCm Compute Profiler) -- see
https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
for details.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --report)
      REPORT="$2"; shift 2 ;;
    --top)
      SELECTION_ARGS=(--top "$2"); shift 2 ;;
    --threshold)
      SELECTION_ARGS=(--threshold "$2"); shift 2 ;;
    --all)
      SELECTION_ARGS=(--all); shift ;;
    --workload-dir)
      WORKLOAD_DIR="$2"; shift 2 ;;
    --all-dispatches)
      ALL_DISPATCHES=1; shift ;;
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

if [[ "$ALL_DISPATCHES" -eq 1 ]]; then
  echo "warning: --all-dispatches profiles every call of each selected kernel, not just the 2nd -- for a kernel called N times this can multiply profiling time by roughly N (on top of rocprof-compute's own multi-pass counter replay). Only use this for a small test case specifically sized for rocprof-compute; for a normal/long-running application, drop this flag and let the default (-d 2, second call only) keep runtime bounded." >&2
fi

WORKLOAD_NAME="$(basename "$WORKLOAD_DIR")"

MPI_ARR=()
if [[ -n "$MPI_STR" ]]; then
  read -ra MPI_ARR <<< "$MPI_STR"
fi
MPI_FORWARD=()
[[ -n "$MPI_STR" ]] && MPI_FORWARD=(--mpi "$MPI_STR")
ALL_DISPATCHES_ARGS=()
[[ "$ALL_DISPATCHES" -eq 1 ]] && ALL_DISPATCHES_ARGS=(--all-dispatches)

if ! command -v rocprof-compute >/dev/null 2>&1; then
  echo "error: 'rocprof-compute' not found on PATH." >&2
  echo "       Load the ROCm module providing rocprofiler-compute (e.g. 'module load rocm') and retry." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: 'python3' not found on PATH -- required to select hotspot kernels." >&2
  exit 1
fi

# MPI multi-rank safety gate. Runs even under --dry-run (both probes below are cheap and
# side-effect-free relative to the real workload: a --version/--help call, and a trivial
# `echo` under the user's own launch command) -- a --dry-run preview should reflect the same
# go/no-go outcome a real run would reach, not silently skip this check.
if [[ -n "$MPI_STR" ]]; then
  ROCPROF_COMPUTE_VERSION="$(rocprof-compute -v 2>&1)"

  CAPABLE=0
  if rocprof-compute profile --help 2>&1 | grep -q '%rank%'; then
    CAPABLE=1
  fi

  # Generic rank-count probe: works for any launcher (mpirun/srun/mpiexec/...), not just
  # mpirun-specific flag parsing -- any MPI launcher spawns N copies of whatever command it's
  # given, so N echoes back means N ranks.
  set +e
  RANK_PROBE_OUTPUT="$("${MPI_ARR[@]}" echo __rank_probe__ 2>/dev/null)"
  set -e
  RANK_COUNT="$(printf '%s\n' "$RANK_PROBE_OUTPUT" | grep -c '__rank_probe__' || true)"

  if [[ "$RANK_COUNT" -eq 0 ]]; then
    echo "error: couldn't determine how many ranks --mpi \"$MPI_STR\" launches (the probe produced no output) -- refusing to guess. Check the launch command, or drop --mpi if your program doesn't actually need it." >&2
    exit 1
  fi

  if [[ "$RANK_COUNT" -gt 1 && "$CAPABLE" -eq 0 ]]; then
    echo "error: MPI is not supported with this version of rocprof-compute ($ROCPROF_COMPUTE_VERSION; detected $RANK_COUNT ranks; no %rank% output-isolation support found in 'rocprof-compute profile --help') -- rerun with a single-rank launch, or upgrade rocprof-compute." >&2
    exit 1
  fi
fi

if [[ -z "$REPORT" ]]; then
  SCAN_CMD=("$GPU_LAUNCHER" --no-summary "${MPI_FORWARD[@]}" "${SELECTION_ARGS[@]}" -o "$OUTPUT_DIR" -- "$@")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "would run:"
    printf '  %q ' "${SCAN_CMD[@]}"
    echo
  else
    "${SCAN_CMD[@]}"
  fi
fi

# Resolve kernel names. A --report is just a file read -- safe to do for real even under
# --dry-run. Without one, the data only exists if the scan above actually ran, so under
# --dry-run (where it didn't) the profile preview below uses a placeholder instead.
KERNELS_OUTPUT=""
RESOLVED_REAL_KERNELS=0
if [[ -n "$REPORT" ]]; then
  RESOLVED_REAL_KERNELS=1
  set +e
  KERNELS_OUTPUT="$(python3 "$SELECTOR" --report "$REPORT" "${ALL_DISPATCHES_ARGS[@]}")"
  SELECTOR_EXIT=$?
  set -e
  [[ "$SELECTOR_EXIT" -eq 0 ]] || exit "$SELECTOR_EXIT"
elif [[ "$DRY_RUN" -ne 1 ]]; then
  RESOLVED_REAL_KERNELS=1
  set +e
  KERNELS_OUTPUT="$(python3 "$SELECTOR" --output-dir "$OUTPUT_DIR" "${SELECTION_ARGS[@]}" "${ALL_DISPATCHES_ARGS[@]}")"
  SELECTOR_EXIT=$?
  set -e
  [[ "$SELECTOR_EXIT" -eq 0 ]] || exit "$SELECTOR_EXIT"
fi

KERNELS=()
if [[ -n "$KERNELS_OUTPUT" ]]; then
  while IFS= read -r kernel; do
    KERNELS+=("$kernel")
  done <<< "$KERNELS_OUTPUT"
fi

if [[ "$RESOLVED_REAL_KERNELS" -eq 1 && ${#KERNELS[@]} -eq 0 ]]; then
  echo "error: no hotspot kernel is called more than once in this run -- nothing eligible to profile with -d 2 (rerun with --all-dispatches to include single-call kernels too)" >&2
  exit 1
fi

DISPATCH_ARGS=()
[[ "$ALL_DISPATCHES" -ne 1 ]] && DISPATCH_ARGS=(-d 2)

if [[ ${#KERNELS[@]} -gt 0 ]]; then
  PROFILE_CMD=("${MPI_ARR[@]}" rocprof-compute profile -n "$WORKLOAD_NAME" -p "$WORKLOAD_DIR" \
               -k "${KERNELS[@]}" "${DISPATCH_ARGS[@]}" -- "$@")
else
  PROFILE_CMD=("${MPI_ARR[@]}" rocprof-compute profile -n "$WORKLOAD_NAME" -p "$WORKLOAD_DIR" \
               -k "<resolved from the scan above -- not run under --dry-run>" "${DISPATCH_ARGS[@]}" -- "$@")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would run:"
  printf '  %q ' "${PROFILE_CMD[@]}"
  echo
  echo "would then run:"
  printf '  %q ' rocprof-compute analyze -p "$WORKLOAD_DIR"
  printf '| tee %q\n' "$WORKLOAD_DIR/analysis.txt"
  exit 0
fi

mkdir -p "$WORKLOAD_DIR"

set +e
"${PROFILE_CMD[@]}"
APP_EXIT=$?
set -e

if [[ "$RUN_SUMMARY" -eq 1 ]]; then
  rocprof-compute analyze -p "$WORKLOAD_DIR" | tee "$WORKLOAD_DIR/analysis.txt" || \
    echo "warning: rocprof-compute analyze failed; profiling data is still in $WORKLOAD_DIR" >&2
fi

exit "$APP_EXIT"
