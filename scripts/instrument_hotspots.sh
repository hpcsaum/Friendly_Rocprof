#!/usr/bin/env bash
# Builds a rocprof-sys-instrument binary rewrite that instruments only the
# CPU functions already identified as hotspots by profile_hotspots.sh (or a
# saved hotspots.txt report), instead of every function in the binary --
# then, in "trace" mode, runs that rewritten binary to produce a full trace.
#
# Two modes, one script, sharing the same build pipeline:
#   instrument_hotspots.sh instrument [options] -- <executable> [args...]
#   instrument_hotspots.sh trace      [options] -- <executable> [args...]
# "trace" runs every step "instrument" runs, then also runs the result --
# it is never a separate pipeline that consumes a prior "instrument" run's
# output; each invocation builds its own instrumented binary.
#
# MPI is driven by this script itself via --mpi "<launch command>" (e.g.
# --mpi "mpirun -np 4") -- same convention as every other launcher in this
# project: the binary rewrite must happen exactly once, never once per rank,
# so this script expects to be invoked exactly once per use (e.g. from inside
# an existing SLURM/PBS allocation), and decides internally where the given
# MPI command applies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOTSPOTS_LAUNCHER="$SCRIPT_DIR/profile_hotspots.sh"
SELECTOR="$SCRIPT_DIR/../postprocess/tools/select_instrumented_functions.py"

usage_top() {
  cat <<'EOF'
Usage: instrument_hotspots.sh <instrument|trace> [options] -- <executable> [args...]

Instrumenting every function in a binary to trace it is slow and distorts
the very timing you're trying to measure. This tool instruments ONLY the
functions that a previous profile_hotspots.sh run (or one it runs for you)
already flagged as hotspots -- a much smaller, much lower-overhead
experiment, aimed at getting a detailed trace of just the parts that
matter.

Two modes, sharing the same build:
  instrument   build the instrumented binary, then stop.
  trace        do the same build, then immediately run it to produce a
               full trace (kernels and MPI/OpenMP activity included too).

Run 'instrument_hotspots.sh instrument -h' or '... trace -h' for that
mode's own options.

Under the hood, this uses AMD's rocprof-sys-instrument and rocprof-sys-run
-- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/docs-7.0.2/how-to/instrumenting-rewriting-binary-application.html
for details.
EOF
}

usage_instrument() {
  cat <<'EOF'
Usage: instrument_hotspots.sh instrument [options] -- <executable> [args...]

Builds a rocprof-sys-instrument binary rewrite ("<executable>.inst" by
default) that instruments ONLY the hotspot functions found by
profile_hotspots.sh (run automatically with a 1% threshold unless you pass
--report), instead of every function in the binary -- much lower overhead
than a full instrumentation run. This mode only builds the binary; it
never runs it. Use 'trace' mode instead if you also want to run it.

If your program needs MPI to run at all, pass --mpi "<launch command>"
(e.g. --mpi "mpirun -np 4") -- it's used for the profiling run only; the
rewrite itself is always a single, un-prefixed process, since it doesn't
execute your program at all.

Under the hood, this uses AMD's rocprof-sys-instrument -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/docs-7.0.2/how-to/instrumenting-rewriting-binary-application.html
for details.

Options:
  -o, --output-dir DIR    directory for the auto-profiling scan (default: instrument_hotspots-scan-<timestamp>)
  --report FILE            reuse an existing hotspots.txt instead of auto-profiling
  --top N                  hotspot functions to select (last of --top/--threshold/--all wins)
  --threshold PCT          only select functions at or above PCT% of total runtime (default: 1)
  --all                    select every function found, no truncation
  --no-summary             skip writing the auto-profiling run's own hotspots.txt and calltree.txt
  --unfiltered             select hotspot functions by inclusive (total) time instead of self
                           time -- the old behavior, which can pick a function that just calls
                           other functions rather than one that does real work
  --max-depth N            truncate the auto-profiling run's own calltree.txt at this depth
                           (default: unlimited)
  --show-gpu-api           include GPU-API/offload-runtime noise in the auto-profiling run's calltree.txt
  --show-rocprofsys-internals  include rocprof-sys's own instrumentation/GOTCHA frames instead of
                           splicing them out of the auto-profiling run's calltree.txt
  --show-mpi-internals     include MPI library internals below the first MPI frame in the
                           auto-profiling run's calltree.txt instead of collapsing them
  --show-compiler-runtime  include compiler-runtime allocator/intrinsic helper noise in the
                           auto-profiling run's calltree.txt
  --show-all-internals     shorthand for all four --show-* flags above at once
  --mpi "<launch cmd>"     MPI launch command to prefix the auto-profiling run with (e.g. "mpirun -np 4")
  --out-binary PATH        path for the instrumented binary (default: <executable>.inst)
  --dry-run                print what would run, don't execute
  -h, --help                show this help
EOF
}

usage_trace() {
  cat <<'EOF'
Usage: instrument_hotspots.sh trace [options] -- <executable> [args...]

Does everything 'instrument' mode does (build a rocprof-sys-instrument
binary rewrite covering only the hotspot functions found by
profile_hotspots.sh, or a --report you already have), then immediately
runs the result to produce a full trace. The trace also automatically
captures GPU kernel launches/copies, OpenMP regions, and MPI calls -- not
just the hotspot functions themselves -- so kernels and communication
still show up. OpenMP tracing needs an OpenMP runtime that supports OMPT;
MPI tracing needs a rocprof-sys build with MPI support -- a trace missing
one of those isn't necessarily a bug in this script.

If your program needs MPI to run at all, pass --mpi "<launch command>"
(e.g. --mpi "mpirun -np 4") -- it's reused for both the profiling run and
the final trace run; the binary rewrite itself is always un-prefixed.

Under the hood, this uses AMD's rocprof-sys-instrument and rocprof-sys-run
-- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/docs-7.0.2/how-to/instrumenting-rewriting-binary-application.html
and
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/docs-7.0.2/how-to/configuring-runtime-options.html
for details.

Options:
  -o, --output-dir DIR       directory for the auto-profiling scan (default: instrument_hotspots-scan-<timestamp>)
  --report FILE               reuse an existing hotspots.txt instead of auto-profiling
  --top N                     hotspot functions to select (last of --top/--threshold/--all wins)
  --threshold PCT             only select functions at or above PCT% of total runtime (default: 1)
  --all                       select every function found, no truncation
  --no-summary                 skip writing the auto-profiling run's own hotspots.txt and calltree.txt
  --unfiltered                 select hotspot functions by inclusive (total) time instead of
                               self time -- the old behavior, which can pick a function that
                               just calls other functions rather than one that does real work
  --max-depth N                truncate the auto-profiling run's own calltree.txt at this depth
                               (default: unlimited)
  --show-gpu-api               include GPU-API/offload-runtime noise in the auto-profiling run's calltree.txt
  --show-rocprofsys-internals  include rocprof-sys's own instrumentation/GOTCHA frames instead of
                               splicing them out of the auto-profiling run's calltree.txt
  --show-mpi-internals         include MPI library internals below the first MPI frame in the
                               auto-profiling run's calltree.txt instead of collapsing them
  --show-compiler-runtime      include compiler-runtime allocator/intrinsic helper noise in the
                               auto-profiling run's calltree.txt
  --show-all-internals         shorthand for all four --show-* flags above at once
  --mpi "<launch cmd>"        MPI launch command, reused for the profiling run and the trace run
  --out-binary PATH            path for the instrumented binary (default: <executable>.inst)
  --trace-output-dir DIR      directory for the trace output (default: instrument_hotspots-trace-output-<timestamp>)
  --dry-run                    print what would run, don't execute
  -h, --help                    show this help
EOF
}

MODE="${1:-}"
case "$MODE" in
  instrument|trace)
    shift
    ;;
  -h|--help|"")
    usage_top
    if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then exit 0; else exit 1; fi
    ;;
  *)
    echo "error: unknown mode '$MODE' (expected 'instrument' or 'trace')" >&2
    usage_top >&2
    exit 1
    ;;
esac

usage() {
  if [[ "$MODE" == "trace" ]]; then usage_trace; else usage_instrument; fi
}

# Saved before option parsing consumes "$@" -- lets both modes suggest a
# ready-to-paste command later by just swapping the mode word, instead of
# manually reconstructing every flag.
ORIGINAL_ARGS=("$@")

RUN_TS="$(date +%F_%H.%M.%S)"
OUTPUT_DIR="instrument_hotspots-scan-$RUN_TS"
TRACE_OUTPUT_DIR="instrument_hotspots-trace-output-$RUN_TS"
REPORT=""
SELECTION_ARGS=()
NO_SUMMARY=0
UNFILTERED=0
MPI_STR=""
OUT_BINARY=""
DRY_RUN=0
CALLTREE_ARGS=()

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
    --no-summary)
      NO_SUMMARY=1; shift ;;
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
    --out-binary)
      OUT_BINARY="$2"; shift 2 ;;
    --trace-output-dir)
      TRACE_OUTPUT_DIR="$2"; shift 2 ;;
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
  echo "error: no executable given after --" >&2
  usage >&2
  exit 1
fi

BINARY="$1"
APP_ARGS=("$@")

if [[ ! -e "$BINARY" ]]; then
  echo "error: no such file: $BINARY" >&2
  exit 1
fi
if [[ ! -x "$BINARY" ]]; then
  echo "error: not executable: $BINARY" >&2
  exit 1
fi

if [[ -z "$OUT_BINARY" ]]; then
  OUT_BINARY="$BINARY.inst"
fi

# Nothing has been explicitly chosen -- default to the 1% threshold this
# whole tool is built around, not tools 1-3's own top-20 default.
if [[ ${#SELECTION_ARGS[@]} -eq 0 ]]; then
  SELECTION_ARGS=(--threshold 1)
fi

MPI_ARR=()
if [[ -n "$MPI_STR" ]]; then
  read -ra MPI_ARR <<< "$MPI_STR"
fi
MPI_FORWARD=()
[[ -n "$MPI_STR" ]] && MPI_FORWARD=(--mpi "$MPI_STR")
UNFILTERED_ARGS=()
[[ "$UNFILTERED" -eq 1 ]] && UNFILTERED_ARGS=(--unfiltered)

if ! command -v rocprof-sys-instrument >/dev/null 2>&1; then
  echo "error: 'rocprof-sys-instrument' not found on PATH." >&2
  echo "       Load the ROCm module providing rocprofiler-systems (e.g. 'module load rocm') and retry." >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "error: 'python3' not found on PATH -- required to select hotspot functions." >&2
  exit 1
fi
if [[ "$MODE" == "trace" ]]; then
  if ! command -v rocprof-sys-run >/dev/null 2>&1; then
    echo "error: 'rocprof-sys-run' not found on PATH." >&2
    echo "       Load the ROCm module providing rocprofiler-systems (e.g. 'module load rocm') and retry." >&2
    exit 1
  fi
fi

# Step: obtain hotspot data, unless a --report was given to reuse instead.
if [[ -z "$REPORT" ]]; then
  NO_SUMMARY_ARGS=()
  [[ "$NO_SUMMARY" -eq 1 ]] && NO_SUMMARY_ARGS=(--no-summary)
  PROFILE_CMD=("$HOTSPOTS_LAUNCHER" "${MPI_FORWARD[@]}" "${SELECTION_ARGS[@]}" "${UNFILTERED_ARGS[@]}" "${CALLTREE_ARGS[@]}" "${NO_SUMMARY_ARGS[@]}" -o "$OUTPUT_DIR" -- "${APP_ARGS[@]}")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "would run:"
    printf '  %q ' "${PROFILE_CMD[@]}"
    echo
  else
    "${PROFILE_CMD[@]}"
  fi
fi

# Step: resolve hotspot function labels/regexes. A --report is just a file
# read -- safe to do for real even under --dry-run. Without one, the data
# only exists if the profiling step above actually ran, so under --dry-run
# (where it didn't) the rewrite preview below uses a placeholder instead.
PAIRS_OUTPUT=""
if [[ -n "$REPORT" ]]; then
  set +e
  PAIRS_OUTPUT="$(python3 "$SELECTOR" --report "$REPORT")"
  SELECTOR_EXIT=$?
  set -e
  [[ "$SELECTOR_EXIT" -eq 0 ]] || exit "$SELECTOR_EXIT"
elif [[ "$DRY_RUN" -ne 1 ]]; then
  set +e
  PAIRS_OUTPUT="$(python3 "$SELECTOR" --output-dir "$OUTPUT_DIR/rocprof-sys" --gpu-output-dir "$OUTPUT_DIR/rocprofv3" "${SELECTION_ARGS[@]}" "${UNFILTERED_ARGS[@]}")"
  SELECTOR_EXIT=$?
  set -e
  [[ "$SELECTOR_EXIT" -eq 0 ]] || exit "$SELECTOR_EXIT"
fi

LABELS=()
REGEXES=()
if [[ -n "$PAIRS_OUTPUT" ]]; then
  while IFS=$'\t' read -r label regex; do
    LABELS+=("$label")
    REGEXES+=("$regex")
  done <<< "$PAIRS_OUTPUT"
fi

# The breakdown of raw hotspots vs. GPU-kernel-owner/ancestor additions is already printed to
# stderr by $SELECTOR itself as it runs -- this is just the final, combined list to instrument.
if [[ ${#LABELS[@]} -gt 0 ]]; then
  echo "functions to instrument (${#LABELS[@]}):"
  for l in "${LABELS[@]}"; do echo "  - $l"; done
fi

# Step: binary rewrite -- always a single, un-prefixed process, regardless
# of --mpi: rewriting the binary has nothing to do with a running MPI job.
REWRITE_CMD=(rocprof-sys-instrument)
if [[ ${#REGEXES[@]} -gt 0 ]]; then
  for r in "${REGEXES[@]}"; do
    REWRITE_CMD+=(-R "$r")
  done
else
  REWRITE_CMD+=(-R "<resolved from the profiling run above -- not run under --dry-run>")
fi
REWRITE_CMD+=(--print-dir "$OUT_BINARY.rocprof-sys-info" -o "$OUT_BINARY" -- "$BINARY")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would run:"
  printf '  %q ' "${REWRITE_CMD[@]}"
  echo
  # instrument mode's preview stops here -- nothing was actually built, so
  # printing a binary path or a lost-function check would be misleading.
  # trace mode falls through to also preview the final run below.
  if [[ "$MODE" == "instrument" ]]; then
    exit 0
  fi
else
  "${REWRITE_CMD[@]}"

  # Non-fatal: warns about any requested function that didn't make it into
  # the binary, using rocprof-sys-instrument's own instrumented.json --
  # never affects this script's exit code either way.
  printf '%s\n' "${LABELS[@]}" | python3 "$SELECTOR" --check-instrumented "$OUT_BINARY.rocprof-sys-info/instrumented.json" || true

  # $OUT_BINARY itself always sits next to the original executable, which isn't
  # necessarily anywhere near $OUTPUT_DIR (e.g. --out-binary pointing elsewhere, or a
  # read-only source tree) -- copy it and its instrumentation-info sidecar into this
  # run's own output directory too, so the binary that produced a given scan/trace is
  # always found alongside it. $OUTPUT_DIR may not exist yet here (e.g. --report skips
  # the profiling step that would otherwise create it).
  mkdir -p "$OUTPUT_DIR"
  COPIED_BINARY="$OUTPUT_DIR/$(basename "$OUT_BINARY")"
  cp "$OUT_BINARY" "$COPIED_BINARY"
  rm -rf "$COPIED_BINARY.rocprof-sys-info"
  cp -r "$OUT_BINARY.rocprof-sys-info" "$COPIED_BINARY.rocprof-sys-info"

  if [[ "$MODE" == "instrument" ]]; then
    echo "wrote instrumented binary: $OUT_BINARY (copy saved under $OUTPUT_DIR)"
    echo "to generate the trace, run:"
    printf '  %q ' "$0" trace "${ORIGINAL_ARGS[@]}"
    echo
    exit 0
  fi
fi

# trace mode only: run the just-built instrumented binary now, in the same
# invocation. Same instrumented binary either way -- these env vars pick
# the full-trace runtime behavior over the build step above.
export ROCPROFSYS_OUTPUT_PATH="$TRACE_OUTPUT_DIR"
export ROCPROFSYS_TRACE=1
export ROCPROFSYS_USE_ROCM=1
export ROCPROFSYS_USE_OMPT=1
export ROCPROFSYS_USE_MPIP=1

RUN_CMD=("${MPI_ARR[@]}" rocprof-sys-run -- "$OUT_BINARY" "${APP_ARGS[@]:1}")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "would export:"
  echo "  ROCPROFSYS_OUTPUT_PATH=$TRACE_OUTPUT_DIR"
  echo "  ROCPROFSYS_TRACE=1"
  echo "  ROCPROFSYS_USE_ROCM=1"
  echo "  ROCPROFSYS_USE_OMPT=1"
  echo "  ROCPROFSYS_USE_MPIP=1"
  echo "would run:"
  printf '  %q ' "${RUN_CMD[@]}"
  echo
  exit 0
fi

mkdir -p "$TRACE_OUTPUT_DIR"

set +e
"${RUN_CMD[@]}"
APP_EXIT=$?
set -e

echo "trace written under $TRACE_OUTPUT_DIR (a Perfetto trace -- view it at ui.perfetto.dev; not parsed by any tool in this project)"
echo "instrumented binary that produced it: $OUT_BINARY (copy saved under $OUTPUT_DIR)"
echo "to regenerate this trace, run:"
printf '  %q ' "$0" trace "${ORIGINAL_ARGS[@]}"
echo

exit "$APP_EXIT"
