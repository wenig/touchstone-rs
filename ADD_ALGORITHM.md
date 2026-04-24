# Contributing an Algorithm

Touchstone-rs is designed as a shared benchmark where researchers contribute streaming anomaly detectors by opening a pull request that adds a new crate under `algorithms/`. Once merged, CI runs the benchmark against every registered algorithm so results are reproducible and directly comparable across submissions.

This guide walks through the contribution workflow. You only need to implement the `Detector` trait, as evaluation, CLI parsing, and reporting are all handled by `touchstone-rs` via the `touchstone_main!` macro. You never write a `main` function yourself.

## 1. Fork and clone

Fork the repository on GitHub, then clone your fork and create a feature branch:

```sh
git clone https://github.com/<your-user>/touchstone-rs.git
cd touchstone-rs
git checkout -b add-my-detector
```

## 2. Create and initialize the crate

```sh
mkdir algorithms/my_detector
cd algorithms/my_detector
cargo init --bin --name my_detector
```

The workspace root (`Cargo.toml`) already globs `algorithms/*`, so the new crate is picked up automatically — no workspace edit needed.

## 3. Add the `touchstone-rs` dependency

From inside `algorithms/my_detector/`:

```sh
cargo add --path ../../touchstone-rs
```

Add any crates your detector needs (e.g. `ndarray`, `rand`) the same way.

## 4. Replace `src/main.rs`

Implement the `Detector` trait and invoke `touchstone_main!`:

```rust
use touchstone_rs::{Detector, touchstone_main};

pub struct MyDetector {
    n_dimensions: usize,
}

impl Detector for MyDetector {
    fn name() -> &'static str {
        "MyDetector"
    }

    fn new(n_dimensions: usize) -> Self {
        Self { n_dimensions }
    }

    fn update(&mut self, point: &[f32]) -> f32 {
        // return an anomaly score; NaN during warmup is fine
        0.0
    }
}

touchstone_main!(MyDetector);
```

`name()` returns the display name used when your detector appears in the results DataFrame and comparison tables.

## 5. Run it locally

From the workspace root:

```sh
cargo run -p my_detector -- --data-dir data
```

`--data-dir` points at a directory of Touchstone CSV datasets. See `README.md` for the dataset format. The run generates a file `touchstone-MyDetector.csv` (using your detector's `name()`) containing results for each dataset, with columns for each metric plus `time_sec`.

## 6. Open a pull request

Push your branch to your fork and open a PR against `develop`. CI runs a **smoke test** on the PR that builds your crate and verifies its runtime stays within the limits of the GitHub-hosted runner. If your detector is too slow on the smoke dataset, the PR is blocked.

The full benchmark and the updated comparison table are produced only **after merge**, so benchmark results are always pinned to code that has passed review.

## Reference: `baseline`

`algorithms/baseline/` is a minimal working example — a random-score detector — useful as a starting template.
