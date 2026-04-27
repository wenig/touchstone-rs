#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <detector-name>"
  exit 1
fi

DETECTOR="$1"
DIR="algorithms/$DETECTOR"

if [ -f "$DIR/Cargo.toml" ]; then
  echo "Building $DETECTOR (Rust) in release mode..."
  cargo build -p "$(basename "$DETECTOR")" --release
  echo "✓ Build successful"
elif [ -f "$DIR/pyproject.toml" ]; then
  echo "Installing $DETECTOR (Python) dependencies..."
  uv sync --project "$DIR"
  echo "✓ Dependencies installed"
else
  echo "Error: $DIR contains neither Cargo.toml nor pyproject.toml"
  exit 1
fi
