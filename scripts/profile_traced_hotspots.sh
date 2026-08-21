#!/usr/bin/env bash
# Runs instrument_hotspots.sh's own "trace" mode (sample to find hotspot functions, instrument
# just those, run the result to produce a raw Perfetto trace), then goes further: converts that
# trace to the flat CSV files this project's trace-based tools read
# (postprocess/tools/convert_trace_to_csv.py), then builds a combined hotspots report and a call
# tree from it (postprocess/tools/extract_trace_hotspots.py/extract_trace_calltree.py) --
# everything instrument_hotspots.sh trace already gives you, plus the reports on top, in one
# invocation. instrument_hotspots.sh itself is unchanged and still the right tool for someone who
# just wants the raw trace (e.g. to open in Perfetto's own UI) and nothing else.
#
# Two selection concepts share flag names with instrument_hotspots.sh's own flags but mean
# different things here, so they're deliberately split into two namespaces: bare
# --top/--threshold/--all/--unfiltered control the FINAL report (forwarded to
# extract_trace_hotspots.py); --instrument-top/--instrument-threshold/--instrument-all/
# --instrument-unfiltered control which functions get INSTRUMENTED in the first place (forwarded
# to instrument_hotspots.sh's own same-named flags) -- for users who want those to diverge from
# the default, expected to be rare enough that the un-prefixed, more convenient names should mean
# the thing most invocations actually care about (the report they'll read), not the one-time
# instrumentation-selection choice.
#
# MPI is driven by this script itself via --mpi "<launch command>" (e.g. --mpi "mpirun -np 4"),
# forwarded straight through to instrument_hotspots.sh -- same convention as every other launcher
# in this project.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTRUMENT_LAUNCHER="$SCRIPT_DIR/instrument_hotspots.sh"
CONVERTER="$SCRIPT_DIR/../postprocess/tools/convert_trace_to_csv.py"
HOTSPOTS_EXTRACTOR="$SCRIPT_DIR/../postprocess/tools/extract_trace_hotspots.py"
CALLTREE_EXTRACTOR="$SCRIPT_DIR/../postprocess/tools/extract_trace_calltree.py"

usage() {
  cat <<'EOF'
Usage: profile_traced_hotspots.sh [options] -- <executable> [args...]
       profile_traced_hotspots.sh --trace-report DIR [options]

Does everything 'instrument_hotspots.sh trace' does (sample to find hotspot
functions, instrument just those, run the result to produce a full trace),
then converts that trace and builds an actual hotspots.txt + calltree.txt
from it -- the "just give me the reports" version of instrument_hotspots.sh,
for someone who doesn't want to run the conversion and report tools by hand
afterward. Use instrument_hotspots.sh directly instead if you only want the
raw trace (e.g. to open in Perfetto's own UI).

Two independent selection concepts happen to share flag names with
instrument_hotspots.sh's own flags: bare --top/--threshold/--all/--unfiltered
control the FINAL report (how many entries appear in hotspots.txt, and
whether by self or inclusive time); --instrument-top/--instrument-threshold/
--instrument-all/--instrument-unfiltered control which functions get
INSTRUMENTED in the first place, for expert users who want that to diverge
from the report -- left at instrument_hotspots.sh's own default (>=1% of
runtime) if none of the --instrument-* flags are given.

--trace-report DIR skips the sample/instrument/trace steps entirely and
starts straight from an existing trace directory (from a previous run of
this script, a raw instrument_hotspots.sh trace run, or any other source):
no executable is needed in this mode. If DIR already has converted CSV
files, the conversion step is skipped too and only the two report tools run;
otherwise the trace is converted first. Combining --trace-report with
--report/--instrument-top/--instrument-threshold/--instrument-all/
--instrument-unfiltered/--mpi/--out-binary is an error -- none of those
configure a step that runs in this mode.

Under the hood, this uses AMD's rocprof-sys-instrument and rocprof-sys-run,
plus Perfetto's trace_processor_shell (via convert_trace_to_csv.py -- see
that tool's own --help for how it's located) -- see
https://rocm.docs.amd.com/projects/rocprofiler-systems/en/docs-7.0.2/how-to/instrumenting-rewriting-binary-application.html
and https://perfetto.dev/docs/analysis/trace-processor for details.

Options:
  -o, --output-dir DIR        base directory for this run (default: profile_hotspots_trace-output-<timestamp>);
                               written as DIR/scan (the auto-profiling scan) and DIR/trace (the
                               trace, its converted CSVs, hotspots.txt, and calltree.txt)
  --trace-report DIR           skip straight to an existing trace directory (see above); no
                               executable needed
  --report FILE                 reuse an existing hotspots.txt instead of auto-profiling to select
                               which functions to instrument
  --top N                       entries in the final hotspots report (last of --top/--threshold/--all wins)
  --threshold PCT                only report entries at or above PCT% of total runtime
  --all                          report every entry, no truncation
  --unfiltered                   rank the final report by inclusive (total) time instead of self time
  --instrument-top N              hotspot functions to instrument (last of the --instrument-* trio wins;
                               default: instrument_hotspots.sh's own >=1% threshold)
  --instrument-threshold PCT      only instrument functions at or above PCT% of total runtime
  --instrument-all                instrument every function found, no truncation
  --instrument-unfiltered         select functions to instrument by inclusive time instead of self time
  --no-summary                   skip the conversion and report steps -- just the raw trace + binary
  --max-depth N                   truncate every generated call tree at this depth (default: unlimited)
  --show-gpu-api                  include GPU-API/offload-runtime noise in every generated call tree
  --show-rocprofsys-internals     include rocprof-sys's own instrumentation/GOTCHA frames instead of
                               splicing them out of every generated call tree
  --show-mpi-internals             include MPI library internals below the first MPI frame in every
                               generated call tree instead of collapsing them
  --show-compiler-runtime          include compiler-runtime allocator/intrinsic helper noise in
                               every generated call tree
  --show-all-internals             shorthand for all four --show-* flags above at once
  --mpi "<launch cmd>"            MPI launch command (e.g. "mpirun -np 4"), forwarded to instrument_hotspots.sh
  --out-binary PATH                path for the instrumented binary (default: <executable>.inst)
  --trace-processor PATH           path to Perfetto's trace_processor_shell (or its trace_processor
                               wrapper script) -- see convert_trace_to_csv.py --help for the full
                               resolution order (this flag, then $FRIENDLY_ROCPROF_TRACE_PROCESSOR, then PATH)
  --dry-run                        print what would run, don't execute
  -h, --help                       show this help
EOF
}

RUN_TS="$(date +%F_%H.%M.%S)"
OUTPUT_DIR="profile_hotspots_trace-output-$RUN_TS"
TRACE_REPORT=""
REPORT=""
REPORT_SELECTION_ARGS=()
REPORT_UNFILTERED=0
INSTRUMENT_SELECTION_ARGS=()
INSTRUMENT_UNFILTERED=0
NO_SUMMARY=0
CALLTREE_ARGS=()
MPI_STR=""
OUT_BINARY=""
TRACE_PROCESSOR=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --trace-report)
      TRACE_REPORT="$2"; shift 2 ;;
    --report)
      REPORT="$2"; shift 2 ;;
    --top)
      REPORT_SELECTION_ARGS=(--top "$2"); shift 2 ;;
    --threshold)
      REPORT_SELECTION_ARGS=(--threshold "$2"); shift 2 ;;
    --all)
      REPORT_SELECTION_ARGS=(--all); shift ;;
    --unfiltered)
      REPORT_UNFILTERED=1; shift ;;
    --instrument-top)
      INSTRUMENT_SELECTION_ARGS=(--top "$2"); shift 2 ;;
    --instrument-threshold)
      INSTRUMENT_SELECTION_ARGS=(--threshold "$2"); shift 2 ;;
    --instrument-all)
      INSTRUMENT_SELECTION_ARGS=(--all); shift ;;
    --instrument-unfiltered)
      INSTRUMENT_UNFILTERED=1; shift ;;
    --no-summary)
      NO_SUMMARY=1; shift ;;
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

if [[ -n "$TRACE_REPORT" ]]; then
  for conflict_name in REPORT MPI_STR OUT_BINARY; do
    if [[ -n "${!conflict_name}" ]]; then
      echo "error: --trace-report can't be combined with --report/--mpi/--out-binary -- none of those configure a step that runs in this mode" >&2
      exit 1
    fi
  done
  if [[ ${#INSTRUMENT_SELECTION_ARGS[@]} -gt 0 || "$INSTRUMENT_UNFILTERED" -eq 1 ]]; then
    echo "error: --trace-report can't be combined with --instrument-top/--instrument-threshold/--instrument-all/--instrument-unfiltered -- no instrumentation happens in this mode" >&2
    exit 1
  fi
else
  if [[ $# -eq 0 ]]; then
    echo "error: no executable given after --" >&2
    usage >&2
    exit 1
  fi
fi

REPORT_UNFILTERED_ARGS=()
[[ "$REPORT_UNFILTERED" -eq 1 ]] && REPORT_UNFILTERED_ARGS=(--unfiltered)
INSTRUMENT_UNFILTERED_ARGS=()
[[ "$INSTRUMENT_UNFILTERED" -eq 1 ]] && INSTRUMENT_UNFILTERED_ARGS=(--unfiltered)
MPI_FORWARD=()
[[ -n "$MPI_STR" ]] && MPI_FORWARD=(--mpi "$MPI_STR")
TRACE_PROCESSOR_ARGS=()
[[ -n "$TRACE_PROCESSOR" ]] && TRACE_PROCESSOR_ARGS=(--trace-processor "$TRACE_PROCESSOR")

# printf's own %q/%s conversions still run once even with zero arguments (defaulting to an empty
# string) -- calling printf '%q ' "${empty_array[@]}" directly prints a spurious '' instead of
# nothing. Every array-tail printf below goes through this instead.
print_args() {
  if [[ $# -gt 0 ]]; then
    printf '%q ' "$@"
  fi
  return 0
}

# Runs the conversion, then (only if it succeeds -- the two extract tools have nothing to read
# otherwise) the two report tools, against $1 -- non-fatal throughout: the expensive trace/data
# already exists by the time this runs, so a post-processing failure shouldn't look like the
# whole invocation failed, matching profile_hotspots.sh's own precedent for its own report steps.
run_reports() {
  local trace_dir="$1"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "warning: python3 not found, skipping CSV conversion and reports; convert/extract manually later:" >&2
    echo "  python3 $CONVERTER $trace_dir ${TRACE_PROCESSOR_ARGS[*]}" >&2
    echo "  python3 $HOTSPOTS_EXTRACTOR $trace_dir ${REPORT_SELECTION_ARGS[*]} ${REPORT_UNFILTERED_ARGS[*]}" >&2
    echo "  python3 $CALLTREE_EXTRACTOR $trace_dir ${CALLTREE_ARGS[*]}" >&2
    return
  fi
  if python3 "$CONVERTER" "$trace_dir" "${TRACE_PROCESSOR_ARGS[@]}"; then
    python3 "$HOTSPOTS_EXTRACTOR" "$trace_dir" "${REPORT_SELECTION_ARGS[@]}" "${REPORT_UNFILTERED_ARGS[@]}" || \
      echo "warning: hotspots report failed; trace-CSV data is still in $trace_dir" >&2
    python3 "$CALLTREE_EXTRACTOR" "$trace_dir" "${CALLTREE_ARGS[@]}" || \
      echo "warning: calltree report failed; trace-CSV data is still in $trace_dir" >&2
  else
    echo "warning: .proto->CSV conversion failed; skipping hotspots/calltree reports. Raw trace is still in $trace_dir" >&2
  fi
}

if [[ -n "$TRACE_REPORT" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if compgen -G "$TRACE_REPORT"/*-[0-9]*.csv >/dev/null 2>&1; then
      echo "$TRACE_REPORT already has converted CSV files -- would skip conversion"
    else
      echo "would run:"
      printf '  python3 %q %q ' "$CONVERTER" "$TRACE_REPORT"; print_args "${TRACE_PROCESSOR_ARGS[@]}"; echo
    fi
    echo "would run:"
    printf '  python3 %q %q ' "$HOTSPOTS_EXTRACTOR" "$TRACE_REPORT"; print_args "${REPORT_SELECTION_ARGS[@]}" "${REPORT_UNFILTERED_ARGS[@]}"; echo
    printf '  python3 %q %q ' "$CALLTREE_EXTRACTOR" "$TRACE_REPORT"; print_args "${CALLTREE_ARGS[@]}"; echo
    exit 0
  fi

  if compgen -G "$TRACE_REPORT"/*-[0-9]*.csv >/dev/null 2>&1; then
    echo "$TRACE_REPORT already has converted CSV files -- skipping conversion"
    if command -v python3 >/dev/null 2>&1; then
      python3 "$HOTSPOTS_EXTRACTOR" "$TRACE_REPORT" "${REPORT_SELECTION_ARGS[@]}" "${REPORT_UNFILTERED_ARGS[@]}" || \
        echo "warning: hotspots report failed; trace-CSV data is still in $TRACE_REPORT" >&2
      python3 "$CALLTREE_EXTRACTOR" "$TRACE_REPORT" "${CALLTREE_ARGS[@]}" || \
        echo "warning: calltree report failed; trace-CSV data is still in $TRACE_REPORT" >&2
    else
      echo "warning: python3 not found, skipping reports; extract manually later" >&2
    fi
  else
    run_reports "$TRACE_REPORT"
  fi
  exit 0
fi

SCAN_DIR="$OUTPUT_DIR/scan"
TRACE_DIR="$OUTPUT_DIR/trace"

INSTRUMENT_CMD=(
  "$INSTRUMENT_LAUNCHER" trace "${MPI_FORWARD[@]}" "${INSTRUMENT_SELECTION_ARGS[@]}" "${INSTRUMENT_UNFILTERED_ARGS[@]}"
  "${CALLTREE_ARGS[@]}" -o "$SCAN_DIR" --trace-output-dir "$TRACE_DIR"
)
[[ -n "$REPORT" ]] && INSTRUMENT_CMD+=(--report "$REPORT")
[[ -n "$OUT_BINARY" ]] && INSTRUMENT_CMD+=(--out-binary "$OUT_BINARY")
[[ "$NO_SUMMARY" -eq 1 ]] && INSTRUMENT_CMD+=(--no-summary)
[[ "$DRY_RUN" -eq 1 ]] && INSTRUMENT_CMD+=(--dry-run)
INSTRUMENT_CMD+=(-- "$@")

# Under --dry-run, INSTRUMENT_CMD already carries --dry-run through to instrument_hotspots.sh
# itself -- run it for real so IT prints its own accurate "would run:" preview (the actual
# rocprof-sys-instrument/rocprof-sys-run commands, MPI expansion, etc.), the same delegation
# profile_hotspots.sh already uses for its own sub-launcher previews, rather than this script
# trying to reconstruct that detail itself.
"${INSTRUMENT_CMD[@]}"

if [[ "$NO_SUMMARY" -eq 0 ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "would then run:"
    printf '  python3 %q %q ' "$CONVERTER" "$TRACE_DIR"; print_args "${TRACE_PROCESSOR_ARGS[@]}"; echo
    printf '  python3 %q %q ' "$HOTSPOTS_EXTRACTOR" "$TRACE_DIR"; print_args "${REPORT_SELECTION_ARGS[@]}" "${REPORT_UNFILTERED_ARGS[@]}"; echo
    printf '  python3 %q %q ' "$CALLTREE_EXTRACTOR" "$TRACE_DIR"; print_args "${CALLTREE_ARGS[@]}"; echo
  else
    run_reports "$TRACE_DIR"
  fi
fi
