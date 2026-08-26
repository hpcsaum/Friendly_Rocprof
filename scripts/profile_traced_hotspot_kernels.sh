#!/usr/bin/env bash
# Starts from an existing rocprof-sys trace directory (from instrument_hotspots.sh trace or
# profile_traced_hotspots.sh), resolves the GPU kernels with the highest %runtime straight from
# that trace's own recorded data, then builds and runs a rocprof-compute command that profiles
# only those -- specifically each selected kernel's SECOND call (-d 2), skipping the first
# (unrepresentative first-touch/page-fault overhead), then auto-runs rocprof-compute analyze and
# surfaces its own output. This is scripts/profile_hotspot_kernels.sh's "go deep with
# rocprof-compute" pipeline, fed by an already-recorded trace instead of a fresh rocprofv3 scan --
# use this when a trace already exists and paying for a second profiling run just to re-discover
# hot kernels would be redundant.
#
# A kernel dispatched only once has no second call to target and is excluded by default -- pass
# --all-dispatches to include those too (see its own warning below; this can multiply profiling
# time significantly).
#
# MPI: same safety gate as profile_hotspot_kernels.sh -- rocprof-compute's own multi-rank output
# isolation is confirmed ABSENT through rocprofiler-compute 3.4.0 (ROCm 7.2.x), so this script
# probes the installed rocprof-compute's own `profile --help` output for evidence of it, and
# separately determines the actual rank count your --mpi launch command produces, refusing to run
# if more than one rank is detected and no support is confirmed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="$SCRIPT_DIR/../postprocess/tools/convert_trace_to_csv.py"
SELECTOR="$SCRIPT_DIR/../postprocess/tools/select_hotspot_kernels.py"

RUN_TS="$(date +%F_%H.%M.%S)"
OUTPUT_DIR="profile_traced_hotspot_kernels-workload-$RUN_TS"
TRACE_DIR=""
SELECTION_ARGS=()
ALL_DISPATCHES=0
TIME_RANGE=""
RUN_SUMMARY=1
MPI_STR=""
TRACE_PROCESSOR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: profile_traced_hotspot_kernels.sh --trace-dir DIR [options] -- <executable> [args...]

Starts from a trace directory you already have (from instrument_hotspots.sh trace or
profile_traced_hotspots.sh), finds its biggest GPU kernels directly from that trace's own
recorded data, and profiles ONLY those in hardware-counter-level detail -- specifically each
selected kernel's second call, skipping the first (its timing is usually thrown off by one-time
first-touch/page-fault overhead that doesn't represent steady-state cost). Kernels that only run
once are left out entirely by default: there's no second call to target.

This is the "I already have a trace, go deep on what it shows is hot" tool -- use
profile_hotspot_kernels.sh instead if you don't already have a trace and want a fresh scan.

Options:
  --trace-dir DIR          (required) rocprof-sys trace directory to select hotspot kernels from;
                            converted to CSVs first (via convert_trace_to_csv.py) if it doesn't
                            have them already
  -o, --output-dir DIR      directory for rocprof-compute's own output (default:
                            profile_traced_hotspot_kernels-workload-<timestamp>)
  --top N                   hotspot kernels to select (default: 20; last of --top/--threshold/--all wins)
  --threshold PCT           only select kernels at or above PCT% of total GPU kernel time
  --all                     select every kernel found, no truncation
  --all-dispatches          profile EVERY call of each selected kernel instead of just its 2nd --
                            for a kernel called N times this can multiply profiling time by
                            roughly N (on top of rocprof-compute's own multi-pass counter
                            collection). Only use this for a small test case specifically sized
                            for rocprof-compute; for a normal/long-running application, leave
                            this off and let the default (2nd call only) keep runtime bounded.
                            Also includes kernels that only ran once, which are excluded by default.
  --time-range RANGE        only select kernels hot within this window of the trace, e.g.
                            "5:12.5" or "20:" -- see extract_trace_hotspots.py --help for the
                            full RANGE syntax. Default: the whole trace.
  --no-summary              skip auto-running rocprof-compute analyze afterwards
  --mpi "<launch cmd>"      MPI launch command, forwarded to rocprof-compute -- see the MPI note
                            below; this script checks the installed rocprof-compute's own
                            multi-rank support before using this with more than one rank, and
                            refuses to run if it can't confirm it's safe.
  --trace-processor PATH    path to Perfetto's trace_processor_shell (or its trace_processor
                            wrapper script), forwarded to convert_trace_to_csv.py if the trace
                            directory doesn't have converted CSVs yet -- see that tool's own
                            --help for the full resolution order
  --dry-run                 print what would run, don't execute
  -h, --help                show this help

Under the hood, this uses AMD's rocprof-compute (ROCm Compute Profiler) -- see
https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/how-to/profile/mode.html
for details.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace-dir)
      TRACE_DIR="$2"; shift 2 ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --top)
      SELECTION_ARGS=(--top "$2"); shift 2 ;;
    --threshold)
      SELECTION_ARGS=(--threshold "$2"); shift 2 ;;
    --all)
      SELECTION_ARGS=(--all); shift ;;
    --all-dispatches)
      ALL_DISPATCHES=1; shift ;;
    --time-range)
      TIME_RANGE="$2"; shift 2 ;;
    --no-summary)
      RUN_SUMMARY=0; shift ;;
    --mpi)
      MPI_STR="$2"; shift 2 ;;
    --trace-processor)
      TRACE_PROCESSOR="$2"; shift 2 ;;
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

if [[ -z "$TRACE_DIR" ]]; then
  echo "error: --trace-dir is required" >&2
  usage >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "error: no command given after --" >&2
  usage >&2
  exit 1
fi

if [[ "$ALL_DISPATCHES" -eq 1 ]]; then
  echo "warning: --all-dispatches profiles every call of each selected kernel, not just the 2nd -- for a kernel called N times this can multiply profiling time by roughly N (on top of rocprof-compute's own multi-pass counter replay). Only use this for a small test case specifically sized for rocprof-compute; for a normal/long-running application, drop this flag and let the default (-d 2, second call only) keep runtime bounded." >&2
fi

WORKLOAD_NAME="$(basename "$OUTPUT_DIR")"

MPI_ARR=()
if [[ -n "$MPI_STR" ]]; then
  read -ra MPI_ARR <<< "$MPI_STR"
fi
ALL_DISPATCHES_ARGS=()
[[ "$ALL_DISPATCHES" -eq 1 ]] && ALL_DISPATCHES_ARGS=(--all-dispatches)
TIME_RANGE_ARGS=()
[[ -n "$TIME_RANGE" ]] && TIME_RANGE_ARGS=(--time-range "$TIME_RANGE")
TRACE_PROCESSOR_ARGS=()
[[ -n "$TRACE_PROCESSOR" ]] && TRACE_PROCESSOR_ARGS=(--trace-processor "$TRACE_PROCESSOR")

if ! command -v rocprof-compute >/dev/null 2>&1; then
  echo "error: 'rocprof-compute' not found on PATH." >&2
  echo "       Load the ROCm module providing rocprofiler-compute (e.g. 'module load rocm') and retry." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: 'python3' not found on PATH -- required to convert the trace and select hotspot kernels." >&2
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

# convert_trace_to_csv.py is skipped when converted CSVs already exist, matching
# profile_traced_hotspots.sh's own --trace-report DIR convention -- unlike that script's own
# report steps, a conversion failure here is fatal: kernel selection below has nothing to read
# without it.
if compgen -G "$TRACE_DIR"/*-[0-9]*.csv >/dev/null 2>&1; then
  echo "$TRACE_DIR already has converted CSV files -- skipping conversion"
elif [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would run:"
  printf '  python3 %q %q ' "$CONVERTER" "$TRACE_DIR"
  [[ ${#TRACE_PROCESSOR_ARGS[@]} -gt 0 ]] && printf '%q ' "${TRACE_PROCESSOR_ARGS[@]}"
  echo
else
  python3 "$CONVERTER" "$TRACE_DIR" "${TRACE_PROCESSOR_ARGS[@]}"
fi

# Resolving kernel names is a read of the (now-converted) trace CSVs -- safe to do for real even
# under --dry-run, same convention profile_hotspot_kernels.sh uses for its own --report path.
set +e
KERNELS_OUTPUT="$(python3 "$SELECTOR" --trace-dir "$TRACE_DIR" "${SELECTION_ARGS[@]}" "${ALL_DISPATCHES_ARGS[@]}" "${TIME_RANGE_ARGS[@]}")"
SELECTOR_EXIT=$?
set -e
[[ "$SELECTOR_EXIT" -eq 0 ]] || exit "$SELECTOR_EXIT"

KERNELS=()
if [[ -n "$KERNELS_OUTPUT" ]]; then
  while IFS= read -r kernel; do
    KERNELS+=("$kernel")
  done <<< "$KERNELS_OUTPUT"
fi

if [[ ${#KERNELS[@]} -eq 0 ]]; then
  echo "error: no hotspot kernel is called more than once in this trace -- nothing eligible to profile with -d 2 (rerun with --all-dispatches to include single-call kernels too)" >&2
  exit 1
fi

DISPATCH_ARGS=()
[[ "$ALL_DISPATCHES" -ne 1 ]] && DISPATCH_ARGS=(-d 2)

PROFILE_CMD=("${MPI_ARR[@]}" rocprof-compute profile -n "$WORKLOAD_NAME" -p "$OUTPUT_DIR" \
             -k "${KERNELS[@]}" "${DISPATCH_ARGS[@]}" -- "$@")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would run:"
  printf '  %q ' "${PROFILE_CMD[@]}"
  echo
  echo "would then run:"
  printf '  %q ' rocprof-compute analyze -p "$OUTPUT_DIR"
  printf '| tee %q\n' "$OUTPUT_DIR/analysis.txt"
  exit 0
fi

set +e
"${PROFILE_CMD[@]}"
APP_EXIT=$?
set -e

if [[ "$RUN_SUMMARY" -eq 1 ]]; then
  rocprof-compute analyze -p "$OUTPUT_DIR" | tee "$OUTPUT_DIR/analysis.txt" || \
    echo "warning: rocprof-compute analyze failed; profiling data is still in $OUTPUT_DIR" >&2
fi

exit "$APP_EXIT"
