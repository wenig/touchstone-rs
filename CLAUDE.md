# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Touchstone-rs is a Rust library for benchmarking streaming anomaly detectors against labeled time-series datasets. Users implement the `Detector` trait, register detector factories with a `Touchstone` instance, and call `run()` to get a Polars DataFrame of evaluation metrics across all datasets.

## Commands

```sh
cargo build
cargo build --release
cargo test
cargo run --example normal_distribution_detector
cargo doc --open
```

## Architecture

**Core flow:**
1. User creates `Touchstone` (pointing to a data directory)
2. Registers detector factories via `add_detector(name, factory_fn)`
3. Optionally registers custom `Metric` implementations via `add_metric()`
4. Calls `run()` → returns a Polars DataFrame: `[dataset, detector, metric_1…N, time_sec]`

**Key types in `src/lib.rs`:**
- `Detector` trait — one required method: `update(&mut self, point: &[f32]) -> f32`; return `NaN` during warmup
- `Touchstone` struct — owns `DetectorFactory` (boxed closures) and `Vec<Box<dyn Metric>>`
- `FactoryDetector<D, F>` — erases concrete detector type for dynamic dispatch
- Factory pattern: a fresh detector instance is created per dataset via the registered closure

**Data loading (`src/loader.rs`):**
- CSV format: `timestamp (ignored), feature_cols (→ f32), label (→ u8 0/1)`
- `Dataset` stores features in row-major layout: `n_points × n_dims`

**Metrics (`src/metrics/`):**
- `Metric` trait: `fn score(&self, scores: &[f64], labels: &[u8]) -> f64`
- `all_metrics()` returns the default set: ROC-AUC, PR-AUC, Average Precision, Range-PR, VUS-ROC, VUS-PR
- Scores are collected as f32, then minmax-normalized to [0,1] before metric computation
- Range-based metrics (Tatbul et al. NeurIPS 2018) and VUS metrics (Paparrizos et al. PVLDB 2022) are in `range.rs` and `vus.rs`
- Threshold strategies for classification metrics live in `thresholding.rs` (Fixed, Percentile, Sigma)

**Error handling:** `anyhow::Result<T>` throughout; metrics may return `f64::NAN` for undefined cases.

## Dataset Format

200+ CSV benchmark files live in `data/`. Each file: `timestamp, features..., is_anomaly`. The loader skips the timestamp, casts features to f32, and treats the last column as binary labels.

## Adding a Detector

See `examples/normal_distribution_detector.rs` for a complete example. Implement `Detector`, then:

```rust
touchstone.add_detector("my_detector", || MyDetector::new());
```
