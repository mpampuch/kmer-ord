#!/usr/bin/env bash
#
# Peak-RSS memory check for the kmer-ord project pipeline.
#
# Runs the project pipeline across a set of k values, each wrapped in a
# peak-memory measurement, and prints a summary table. Defaults to the
# dev-machine-safe k values (6, 8); set KMERORD_KS="6 10 11" to reproduce the
# original OOM configurations on a big-memory node.
#
# Usage:
#   scripts/mem_test.sh [FASTA] [OUTPUT_DIR]
#
# Env:
#   KMERORD_KS   space-separated k values (default: "6 8")
#   KMERORD_DR   DR methods to pass to --dr (default: "pca" for a fast check;
#                set to the full TESTS.md list for a faithful run)
set -euo pipefail

FASTA="${1:-TEST-DATA/63_Monoraphidiumcircinale.hifi_reads.subsampled.1percent.fasta}"
OUTDIR="${2:-TEST/mem_$(date -u +%Y%m%d_%H%M%S)}"
KS="${KMERORD_KS:-6 8}"
DR="${KMERORD_DR:-pca}"

if [[ ! -f "$FASTA" ]]; then
  echo "ERROR: input FASTA not found: $FASTA" >&2
  exit 1
fi

# Pick the platform's peak-RSS reporter and the units it emits.
#   macOS  /usr/bin/time -l   -> "maximum resident set size" in BYTES
#   Linux  /usr/bin/time -v   -> "Maximum resident set size" in KILOBYTES
case "$(uname -s)" in
  Darwin) TIME_FLAG="-l"; RSS_LABEL="maximum resident set size"; RSS_DIV=$((1024*1024*1024)) ;;
  *)      TIME_FLAG="-v"; RSS_LABEL="Maximum resident set size"; RSS_DIV=$((1024*1024)) ;;
esac

mkdir -p "$OUTDIR"
RESULTS="$OUTDIR/peak_rss.tsv"
printf "k\tpeak_rss_gb\tstatus\n" > "$RESULTS"

for k in $KS; do
  run_out="$OUTDIR/K${k}"
  time_log="$OUTDIR/time_k${k}.log"
  echo ">>> k=${k}  ->  $run_out"

  status="ok"
  /usr/bin/time $TIME_FLAG kmer-ord project \
    --input "$FASTA" \
    --output "$run_out" \
    --threads 1 \
    --kmer "$k" \
    --dr "$DR" \
    --no-tiara \
    --no-matrix-tsv \
    > "$OUTDIR/run_k${k}.log" 2> "$time_log" || status="FAILED(rc=$?)"

  # Extract the peak RSS line emitted by /usr/bin/time.
  rss_raw="$(grep -i "$RSS_LABEL" "$time_log" | grep -oE '[0-9]+' | head -1 || true)"
  if [[ -n "$rss_raw" ]]; then
    peak_gb="$(awk -v v="$rss_raw" -v d="$RSS_DIV" 'BEGIN{printf "%.2f", v/d}')"
  else
    peak_gb="NA"
  fi

  printf "%s\t%s\t%s\n" "$k" "$peak_gb" "$status" | tee -a "$RESULTS"
done

echo
echo "Peak RSS summary ($RESULTS):"
column -t "$RESULTS"
