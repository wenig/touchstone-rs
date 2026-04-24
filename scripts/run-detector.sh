#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <detector-name> <data-dir>"
  exit 1
fi

DETECTOR="$1"
DATA_DIR="$2"

if [ ! -d "$DATA_DIR" ]; then
  echo "Error: test data directory not found: $DATA_DIR"
  exit 1
fi

cargo run -p "$DETECTOR" --release -- --data-dir "$DATA_DIR"

echo "✓ run finished"
