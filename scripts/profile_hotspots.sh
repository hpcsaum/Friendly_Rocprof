#!/usr/bin/env bash
# Run rocprof-sys and rocprofv3 against the same command, one after the other,
# then merge both into one combined hotspots.txt via
# postprocess/extract_hotspots.py. Reuses scripts/profile_CPU_hotspots.sh
# and scripts/profile_GPU_hotspots.sh directly instead of re-implementing their
# dependency checks / flag handling here.
#
# Same MPI convention as the other two launchers: put mpirun/srun *before* this
# script, e.g.:
#
#   mpirun -np 4 scripts/profile_hotspots.sh -o results/run1 -- ./app arg1 arg2
#
# Note: the app runs TWICE here (once under each profiler) with possibly
# different timing/perturbation each time -- that's inherent to combining two
# tools that each wrap a whole process, not something this script can avoid.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACTOR="$SCRIPT_DIR/../postprocess/extract_hotspots.py"
CPU_LAUNCHER="$SCRIPT_DIR/profile_CPU_hotspots.sh"
GPU_LAUNCHER="$SCRIPT_DIR/profile_GPU_hotspots.sh"

OUTPUT_DIR="rocprof-combined-hotspots-output"
RUN_SUMMARY=1
DRY_RUN=0
SELECTION_ARGS=()

usage() {
  cat <<'EOF'
Usage: profile_hotspots.sh [options] -- <command> [args...]

Runs your program TWICE -- once measuring CPU-side time, once measuring
GPU-side time -- then merges both measurements into a single ranked report
telling you, overall, what's worth looking at first: a slow CPU function or
a slow GPU kernel. This is the "just tell me what to look at" version of
profile_CPU_hotspots.sh and profile_GPU_hotspots.sh combined.

Running the program twice means its timing/behavior could differ a little
between the two runs -- only the relative percentages matter here, not
exact times. The two measurements are merged assuming they came from the
same program and the same workload; nothing checks that for you.

Options:
  -o, --output-dir DIR   base output directory (default: rocprof-combined-hotspots-output)
                          split into DIR/rocprof-sys and DIR/rocprofv3
  --top N                 hotspots per table to report (default: 20; last of --top/--threshold/--all wins)
  --threshold PCT         only report entries at or above PCT% of their table's total
  --all                   report every entry, no truncation
  --no-summary            skip auto-running the combined extractor afterwards
  --dry-run               print what would run, don't execute
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

CPU_DIR="$OUTPUT_DIR/rocprof-sys"
GPU_DIR="$OUTPUT_DIR/rocprofv3"

if [[ "$DRY_RUN" -eq 1 ]]; then
  "$CPU_LAUNCHER" --no-summary --dry-run -o "$CPU_DIR" -- "$@"
  "$GPU_LAUNCHER" --no-summary --dry-run -o "$GPU_DIR" -- "$@"
  echo "would then run:"
  printf '  python3 %q %q %q ' "$EXTRACTOR" "$CPU_DIR" "$GPU_DIR"
  printf '%q ' "${SELECTION_ARGS[@]}"
  echo
  exit 0
fi

# No `set +e` around either call, deliberately: if the CPU run fails, abort
# before ever starting the GPU run, rather than combining a failed run's
# partial data with a fresh one. Each sub-launcher already checks its own
# tool is on PATH and fails fast with a clear message if not.
"$CPU_LAUNCHER" --no-summary -o "$CPU_DIR" -- "$@"
"$GPU_LAUNCHER" --no-summary -o "$GPU_DIR" -- "$@"

if [[ "$RUN_SUMMARY" -eq 1 ]]; then
  RANK="${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}"
  if [[ "$RANK" -eq 0 ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python3 "$EXTRACTOR" "$CPU_DIR" "$GPU_DIR" "${SELECTION_ARGS[@]}" || \
        echo "warning: combined hotspots extractor failed; profiling data is still in $CPU_DIR and $GPU_DIR" >&2
    else
      echo "warning: python3 not found, skipping combined summary; run it manually later:" >&2
      echo "  python3 $EXTRACTOR $CPU_DIR $GPU_DIR ${SELECTION_ARGS[*]}" >&2
    fi
  fi
fi
