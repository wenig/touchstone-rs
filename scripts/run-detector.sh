#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <detector-name>"
  exit 1
fi

DETECTOR="$1"
DATA_DIR="tests/fixtures/smoketest"

if [ ! -d "$DATA_DIR" ]; then
  echo "Error: test data directory not found: $DATA_DIR"
  exit 1
fi

echo "Running $DETECTOR with 2-minute timeout..."
timeout 120 cargo run -p "$DETECTOR" --release -- --data-dir "$DATA_DIR"

echo "✓ Smoketest passed"
