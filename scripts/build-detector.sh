#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <detector-name>"
  exit 1
fi

DETECTOR="$1"

echo "Building $DETECTOR in release mode..."
cargo build -p "$DETECTOR" --release

echo "✓ Build successful"
