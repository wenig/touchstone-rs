#!/bin/bash
set -e

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <detector-name> <data-dir>"
  exit 1
fi

DETECTOR="$1"
DATA_DIR="$2"
DIR="algorithms/$DETECTOR"

if [ ! -d "$DATA_DIR" ]; then
  echo "Error: test data directory not found: $DATA_DIR"
  exit 1
fi

if [ -f "$DIR/Cargo.toml" ]; then
  cargo run -p "$(basename "$DETECTOR")" --release -- --data-dir "$DATA_DIR"
elif [ -f "$DIR/pyproject.toml" ]; then
  uv run --project "$DIR" --no-sync python "$DIR/detector.py" --data-dir "$DATA_DIR"
else
  echo "Error: $DIR contains neither Cargo.toml nor pyproject.toml"
  exit 1
fi

echo "✓ run finished"
