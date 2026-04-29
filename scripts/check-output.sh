#!/bin/bash
set -e

OUTPUT=$(ls touchstone-*.csv 2>/dev/null | head -1)

if [ -z "$OUTPUT" ]; then
  echo "Error: no touchstone-*.csv output file found"
  exit 1
fi

echo "Checking $OUTPUT for NaN values..."

HEADER=$(head -1 "$OUTPUT")
ROWS=$(tail -n +2 "$OUTPUT" | wc -l | tr -d ' ')

if [ "$ROWS" -eq 0 ]; then
  echo "Error: $OUTPUT has no data rows"
  exit 1
fi

NAN_COUNT=$(tail -n +2 "$OUTPUT" | grep -ciE '(^|,)nan(,|$)' || true)

if [ "$NAN_COUNT" -gt 0 ]; then
  echo "Error: $OUTPUT contains $NAN_COUNT row(s) with NaN scores"
  tail -n +2 "$OUTPUT" | grep -iE '(^|,)nan(,|$)' | head -10
  exit 1
fi

echo "✓ $OUTPUT looks valid ($ROWS rows, columns: $HEADER)"
