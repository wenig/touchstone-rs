<p align="center">
  <img src="banner.png" />
</p>

[![Live Leaderboard](https://img.shields.io/badge/Live%20Leaderboard-View%20Rankings-%23f97316?logo=github)](https://wenig.github.io/touchstone-rs)
[![crates.io](https://img.shields.io/crates/v/touchstone-rs?label=latest)](https://crates.io/crates/touchstone-rs)
![Tests on main](https://github.com/wenig/touchstone-rs/workflows/Rust/badge.svg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Dependency Status](https://deps.rs/crate/touchstone-rs/0.1.2/status.svg)](https://deps.rs/crate/touchstone-rs/0.1.2)
![Downloads](https://img.shields.io/crates/d/touchstone-rs.svg)

# Touchstone-rs

> **The live leaderboard at [wenig.github.io/touchstone-rs](https://wenig.github.io/touchstone-rs) ranks every contributed algorithm across all 20 benchmark datasets, updated automatically on every merge.** Submit a pull request and your detector appears on the board.

Benchmarking streaming anomaly detectors is tedious: loading dozens of CSV files, wiring up metrics, normalizing scores, and collecting results into a comparable table all take time away from the detector itself. Touchstone-rs handles that scaffolding so you can focus on the algorithm.

Point it at a directory of CSVs, register one or more detectors, call `run()`, and get back a [Polars](https://pola.rs/) DataFrame with one row per `(dataset, detector)` pair covering 14 standard metrics.

Touchstone-rs is designed in the spirit of [TimeEval](https://github.com/TimeEval/TimeEval) \[2\], a Python benchmarking toolkit for time series anomaly detection algorithms. If you are looking for ready-made datasets, the TimeEval evaluation paper \[1\] provides a large collection already formatted for direct use with Touchstone-rs at the [TimeEval Datasets page](https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html).

## Quickstart

Add to `Cargo.toml`:

```toml
[dependencies]
touchstone-rs = "0.1"
```

## Implementing the `Detector` Trait

Every algorithm plugs in through a single trait:

```rust
pub trait Detector: Send {
    fn name() -> &'static str where Self: Sized;
    fn new(n_dimensions: usize) -> Self where Self: Sized;
    fn update(&mut self, point: &[f32]) -> f32;
}
```

- `name()` returns the display name used in the results DataFrame and comparison tables.
- `point` is a slice of `f32` features for the current time step. The length matches the number of feature columns in the dataset.
- Return an **anomaly score** as `f32`. Higher values mean more anomalous.
- Return `f32::NAN` during warmup or whenever a score is not yet meaningful. NaN points are excluded from metric computation.
- Scores are **minmax-normalized** to `[0, 1]` before any metric is computed, so the absolute scale of your scores does not matter.

## Running an Evaluation

```rust,no_run
use std::path::Path;
use touchstone_rs::{Detector, Touchstone};

struct MyDetector { n_dimensions: usize }

impl Detector for MyDetector {
    fn name() -> &'static str { "MyDetector-v1" }

    fn new(n_dimensions: usize) -> Self {
        MyDetector { n_dimensions }
    }

    fn update(&mut self, point: &[f32]) -> f32 {
        // compute and return anomaly score
        0.5
    }
}

fn main() {
    let mut experiment = Touchstone::new(Path::new("data"));

    // `new(n_dimensions)` receives the dataset's feature count at runtime,
    // use it to size internal buffers to match.
    experiment.add_detector::<MyDetector>();

    let df = experiment.run().unwrap();
    println!("{df}");
}
```

## Output DataFrame

`run()` returns a DataFrame with this schema:

| column | type | description |
|---|---|---|
| `dataset` | String | dataset filename (without extension) |
| `detector` | String | name passed to `add_detector` |
| `roc_auc` | f64 | ROC-AUC |
| `pr_auc` | f64 | Precision-Recall AUC |
| `average_precision` | f64 | Average Precision |
| `precision` | f64 | Precision at 90th-percentile threshold |
| `recall` | f64 | Recall at 90th-percentile threshold |
| `f1` | f64 | F1 at 90th-percentile threshold |
| `range_precision` | f64 | Range-based Precision (Tatbul et al., NeurIPS 2018) |
| `range_recall` | f64 | Range-based Recall |
| `range_f_score` | f64 | Range-based F-score |
| `range_auc` | f64 | Range-based AUC |
| `range_pr_vus` | f64 | PR-VUS (Paparrizos et al., PVLDB 2022) |
| `range_roc_vus` | f64 | ROC-VUS |
| `time_sec` | f64 | wall-clock seconds for this detector on this dataset |

If a dataset fails to load or a detector produces only NaN scores, the metric columns for that row contain `NaN`.

## Custom Metrics

If the default metric set does not suit your needs, swap it out entirely by adding metrics before calling `run()`:

```rust,no_run
use std::path::Path;
use touchstone_rs::{Detector, Touchstone};
use touchstone_rs::metrics::{RocAuc, F1Score, SigmaThreshold};

# struct MyDetector { n_dims: usize }
# impl Detector for MyDetector { fn name() -> &'static str { "MyDetector" } fn new(n_dims: usize) -> Self { MyDetector { n_dims } } fn update(&mut self, _: &[f32]) -> f32 { 0.5 } }

let mut experiment = Touchstone::new(Path::new("data"));
experiment.add_detector::<MyDetector>();
experiment.add_metric(RocAuc);
experiment.add_metric(F1Score::new(SigmaThreshold(3.0)));
```

You can also implement `Metric` directly for fully custom scoring:

```rust
use touchstone_rs::metrics::Metric;

struct MyMetric;

impl Metric for MyMetric {
    fn name(&self) -> &str { "my_metric" }
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        // labels: 0 = normal, 1 = anomaly
        // scores: minmax-normalized to [0, 1], NaN already removed
        todo!()
    }
}
```

## Dataset Format

Datasets are CSV files with no assumed column names:

```text
timestamp, feature_1, ..., feature_N, label
2016-04-20 10:35:12, 1.2, 3.4, 0
2016-04-20 10:35:13, 5.6, 7.8, 1
```

- **Column 1**: timestamp | parsed but ignored
- **Columns 2 ... N**: features | cast to `f32`, passed as `point` to `update()`
- **Last column**: binary anomaly label | `0` (normal) or `1` (anomaly)

Touchstone-rs passes every row to `update()` in order, simulating a streaming environment. Each detector gets a fresh instance per dataset.

## Python Bindings (touchstone-py)

`touchstone-py` exposes the same benchmarking engine as a Python package via [PyO3](https://pyo3.rs/), returning results as a [Polars](https://pola.rs/) DataFrame.

### Installation

```sh
pip install touchstone-py
```

Or build from source (requires [maturin](https://github.com/PyO3/maturin) and Rust):

```sh
cd touchstone-py
pip install maturin
maturin develop --release
```

### Implementing a Detector

Subclass `Detector` and implement three members:

```python
from touchstone_py import Detector

class MyDetector(Detector):
    @classmethod
    def name(cls) -> str:
        return "MyDetector-v1"

    def __init__(self, n_dimensions: int) -> None:
        # called once per dataset; size internal state to n_dimensions
        self.n_dimensions = n_dimensions

    def update(self, point: list[float]) -> float:
        # called once per data point in arrival order
        # return a higher score for more anomalous points
        # return float('nan') during warm-up — NaN scores are excluded from metrics
        return 0.5
```

- `name()` — display name shown in the results DataFrame.
- `__init__(n_dimensions)` — receives the dataset's feature count; called once per dataset.
- `update(point)` — receives the current feature vector; return an anomaly score (`float`). Scores are minmax-normalized to `[0, 1]` before metrics are computed, so absolute scale does not matter.

### Running an Evaluation

```python
from pathlib import Path
from touchstone_py import run_touchstone, Detector

class MyDetector(Detector):
    @classmethod
    def name(cls) -> str:
        return "MyDetector-v1"

    def __init__(self, n_dimensions: int) -> None:
        self.window: list[float] = []

    def update(self, point: list[float]) -> float:
        self.window.append(point[0])
        if len(self.window) < 20:
            return float("nan")
        mean = sum(self.window) / len(self.window)
        return abs(point[0] - mean)

df = run_touchstone(Path("data"), [MyDetector])
print(df)
```

`run_touchstone` accepts a directory of CSV datasets and a list of detector *classes* (not instances). It returns a Polars DataFrame with the same schema as the Rust API — one row per `(dataset, detector)` pair with all 14 metrics plus `time_sec`.

## Benchmark Dataset Selection

Evaluating against all 976 TimeEval datasets is expensive and often redundant: many datasets exercise the same failure modes. To address this, we use a data-driven selection procedure to pick a diverse, representative subset.

We extract a feature vector for each dataset describing its performance profile across all 60 algorithms, covering five metrics: ROC-AUC, PR-AUC, Range PR-AUC, Average Precision, and execution time. We then cluster the univariate (902 datasets) and multivariate (74 datasets) subsets independently using k-medoids with k=10 for each, and select the medoid from each cluster as the representative. After filtering for public availability, this yields 19 datasets (10 univariate, 9 multivariate from 6 distinct collections).

The selection provides two key guarantees. First, **diversity**: each chosen dataset represents a distinct performance pattern, avoiding redundant evaluation on algorithmically similar benchmarks. Second, **balance**: both univariate and multivariate time series are covered proportionally despite the class imbalance in the full collection.

We additionally include the CoMuT synthetic dataset, which is purpose-designed to surface correlation anomalies, that is, anomalies invisible in individual channels but visible only through multivariate relationships. This complements TimeEval's focus on point and subsequence anomalies. The resulting 20-dataset benchmark reduces computational cost by 98% while covering both standard detection patterns and correlation-based detection, enabling fast iteration during development while maintaining statistical robustness for final validation.

See [`DATASETS.md`](DATASETS.md) for the complete list of benchmark datasets and their sources.

## Running the Built-in Example

```sh
cargo run --example normal_distribution_detector
```

This runs a rolling z-score detector (window = 20) against all datasets in `data/` and prints the results.

## Contributing an Algorithm

Touchstone-rs accepts new streaming anomaly detectors via pull request. Rust detectors live as their own crate under `algorithms/rust/` and are picked up automatically by the workspace; Python detectors live under `algorithms/python/`. See [`ADD_ALGORITHM.md`](ADD_ALGORITHM.md) for the step-by-step workflow (fork, implement, open PR) and how CI validates submissions.

## References

If you use Touchstone-rs or the TimeEval dataset collection in your work, please cite:

**\[1\] Dataset collection and evaluation methodology**
```bibtex
@article{SchmidlEtAl2022Anomaly,
  title = {Anomaly Detection in Time Series: A Comprehensive Evaluation},
  author = {Schmidl, Sebastian and Wenig, Phillip and Papenbrock, Thorsten},
  date = {2022},
  journaltitle = {Proceedings of the VLDB Endowment (PVLDB)},
  volume = {15},
  number = {9},
  pages = {1779--1797},
  doi = {10.14778/3538598.3538602}
}
```

**\[2\] TimeEval benchmarking toolkit**
```bibtex
@article{WenigEtAl2022TimeEval,
  title = {TimeEval: A Benchmarking Toolkit for Time Series Anomaly Detection Algorithms},
  author = {Wenig, Phillip and Schmidl, Sebastian and Papenbrock, Thorsten},
  date = {2022},
  journaltitle = {Proceedings of the VLDB Endowment (PVLDB)},
  volume = {15},
  number = {12},
  pages = {3678--3681},
  doi = {10.14778/3554821.3554873}
}
```

**\[3\] Touchstone-rs**
```bibtex
@software{TouchstoneRs,
  title = {Touchstone-rs: A Rust Library for Benchmarking Streaming Anomaly Detectors},
  author = {Wenig, Phillip},
  date = {2026},
  url = {https://github.com/wenig/touchstone-rs}
}
```
