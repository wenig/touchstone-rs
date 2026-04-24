#!/bin/bash
set -e

# Get list of changed files compared to main
CHANGED_FILES=$(git diff --name-only origin/main...HEAD)

# Extract detector directories from algorithms/
DETECTORS=$(echo "$CHANGED_FILES" | grep -o '^algorithms/[^/]*' | sort -u | sed 's|algorithms/||')

# Count detectors
COUNT=$(echo "$DETECTORS" | grep -c . || true)

if [ "$COUNT" -eq 0 ]; then
  echo "Error: no algorithm changes found in PR"
  exit 1
fi

if [ "$COUNT" -gt 1 ]; then
  echo "Error: multiple algorithms changed ($COUNT). Only one detector per PR allowed."
  echo "Changed detectors:"
  echo "$DETECTORS" | sed 's/^/  - /'
  exit 1
fi

# Output the detector name
echo "$DETECTORS"
