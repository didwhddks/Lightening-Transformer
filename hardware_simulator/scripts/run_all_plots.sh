#!/usr/bin/env bash
# Run donut plot for every dota_power.csv under results/area_power_all
# Output: dota_power.png saved in the SAME folder as the csv

set -euo pipefail

ROOT_DIR="${1:-../results/area_power_all}"
PLOT_PY="${2:-./plot.py}"   # pass a path if plot.py is elsewhere

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PLOT_PY" ]]; then
  echo "ERROR: plot.py not found: $PLOT_PY" >&2
  exit 1
fi

# Find all target CSVs (skip the top-level file if any; only those in subfolders)
mapfile -t CSV_LIST < <(find "$ROOT_DIR" -type f -name "dota_power.csv" | sort)

if [[ ${#CSV_LIST[@]} -eq 0 ]]; then
  echo "No dota_power.csv found under: $ROOT_DIR" >&2
  exit 1
fi

echo "Found ${#CSV_LIST[@]} csv files."
FAIL=0

for csv in "${CSV_LIST[@]}"; do
  dir="$(dirname "$csv")"
  out_prefix="$dir/dota_power"

  echo "[RUN] $csv -> ${out_prefix}.png"
  if ! python3 "$PLOT_PY" "$csv" "$out_prefix"; then
    echo "[FAIL] $csv" >&2
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "Done with errors." >&2
  exit 1
fi

echo "Done."