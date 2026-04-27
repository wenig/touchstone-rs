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

## Adding a Detector (library API)

See `examples/normal_distribution_detector.rs` for a complete example. Implement `Detector`, then:

```rust
touchstone.add_detector("my_detector", || MyDetector::new());
```

## Adding an Algorithm (contribution workflow)

Two languages are supported. Rust detectors live under `algorithms/rust/`; the workspace root `Cargo.toml` globs `algorithms/rust/*` so no workspace edit is needed. Python detectors live under `algorithms/python/`, which is outside the Cargo workspace glob.

### Rust

**Step 1: create the crate**

```sh
mkdir algorithms/rust/my_detector
cd algorithms/rust/my_detector
cargo init --bin --name my_detector
```

**Step 2: add dependencies**

From inside `algorithms/rust/my_detector/`:

```sh
cargo add --path ../../touchstone-rs
# add any other crates the detector needs, e.g. ndarray, rand
```

**Step 3: implement `src/main.rs`**

Replace the generated `main.rs` with a `Detector` impl and the `touchstone_main!` invocation. Do not write a `main` function yourself; the macro provides one.

```rust
use touchstone_rs::{Detector, touchstone_main};

pub struct MyDetector { n_dimensions: usize }

impl Detector for MyDetector {
    fn name() -> &'static str { "MyDetector" }
    fn new(n_dimensions: usize) -> Self { Self { n_dimensions } }
    fn update(&mut self, point: &[f32]) -> f32 {
        // return anomaly score; NaN during warmup is fine
        0.0
    }
}

touchstone_main!(MyDetector);
```

`name()` is the display name in the results DataFrame and leaderboard. `n_dimensions` reflects the dataset's feature count and is provided at runtime; use it to size internal buffers.

**Step 4: verify locally**

```sh
cargo run -p my_detector -- --data-dir data
```

This writes `touchstone-MyDetector.csv` with one row per dataset covering all metrics plus `time_sec`.

Use `algorithms/rust/baseline/` as a minimal reference implementation.

### Python

**Step 1: create the detector directory**

```sh
mkdir algorithms/python/my_detector
```

**Step 2: declare dependencies in `pyproject.toml`**

```toml
[project]
name = "my_detector"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["touchstone-py"]
```

**Step 3: implement `detector.py`**

```python
from touchstone_py import Detector, run_cli

class MyDetector(Detector):
    @classmethod
    def name(cls) -> str:
        return "MyDetector"

    def __init__(self, n_dimensions: int) -> None:
        self.n_dimensions = n_dimensions

    def update(self, point: list[float]) -> float:
        return 0.0

if __name__ == "__main__":
    run_cli(MyDetector)
```

**Step 4: verify locally**

```sh
uv run --project algorithms/python/my_detector python algorithms/python/my_detector/detector.py --data-dir data
```

### Opening a pull request

Open a PR against `develop`. CI runs a smoke test that builds/installs the detector and checks runtime. The full benchmark runs only after merge.
