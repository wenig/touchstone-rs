//! Touchstone — a streaming anomaly-detection benchmark framework.
//!
//! Implement the [`Detector`] trait, register it with [`Touchstone`], and call
//! [`Touchstone::run`] to evaluate it against a directory of CSV datasets.
//! Results are returned as a [`polars::prelude::DataFrame`] with one row per
//! `dataset × detector` and one column per metric.

pub mod loader;
pub mod metrics;

pub use anyhow::Result;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use loader::Dataset;
use metrics::{Metric, all_metrics, minmax_normalize};
use polars::io::SerWriter;
use polars::prelude::{Column, CsvWriter, DataFrame};
use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Streaming detector interface used by Touchstone during evaluation.
///
/// `update` is called once per point and should return an anomaly score.
/// Returning `NaN` is allowed (for example during warmup).
pub trait Detector: Send {
    /// Display name used in the results DataFrame and comparison tables.
    fn name() -> &'static str
    where
        Self: Sized;
    /// Initialize a new detector given the dimensionality of the stream.
    fn new(n_dimensions: usize) -> Self
    where
        Self: Sized;
    /// Updates detector state with the next point and returns a score.
    fn update(&mut self, point: &[f32]) -> f32;
}

/// Factory for creating fresh detector instances with a specific dimensionality.
pub trait DetectorFactory: Send {
    /// Returns the display name of this detector.
    fn name(&self) -> String;
    /// Creates a new detector instance configured for the given dimensionality.
    fn create(&self, n_dims: usize) -> Box<dyn Detector>;
}

/// Erases the concrete detector type for dynamic dispatch.
struct FactoryDetector<D> {
    /// Marker binding this factory to the detector type `D`.
    _detector: PhantomData<D>,
}

impl<D> DetectorFactory for FactoryDetector<D>
where
    D: Detector + 'static,
{
    fn name(&self) -> String {
        D::name().to_string()
    }

    fn create(&self, n_dims: usize) -> Box<dyn Detector> {
        Box::new(D::new(n_dims))
    }
}

/// Evaluation runner for streaming detectors on Touchstone datasets.
pub struct Touchstone {
    /// Registered detector constructors.
    detector_factories: Vec<Box<dyn DetectorFactory>>,
    /// Metrics used for scoring. Empty means "use defaults at run time".
    metrics: Vec<Box<dyn Metric>>,
    /// Directory containing CSV datasets.
    data_dir: PathBuf,
}

impl Touchstone {
    /// Creates a new evaluation runner for datasets under `data_dir`.
    pub fn new(data_dir: &Path) -> Self {
        Self {
            detector_factories: Vec::new(),
            metrics: Vec::new(),
            data_dir: data_dir.into(),
        }
    }

    /// Registers a detector type.
    ///
    /// The display name comes from `D::name()`. A fresh instance is built per
    /// dataset via `D::new(n_dimensions)`.
    pub fn add_detector<D>(&mut self)
    where
        D: Detector + 'static,
    {
        let detector_factory = FactoryDetector::<D> {
            _detector: PhantomData,
        };
        self.detector_factories.push(Box::new(detector_factory));
    }

    /// Registers a dynamic detector factory.
    ///
    /// Use this when the detector type is not known at compile time (e.g. Python detectors).
    pub fn add_detector_factory(&mut self, factory: Box<dyn DetectorFactory>) {
        self.detector_factories.push(factory);
    }

    /// Adds a custom metric used for scoring.
    ///
    /// If no metrics are added, the default metric set is used.
    pub fn add_metric<M>(&mut self, metric: M)
    where
        M: Metric + 'static,
    {
        self.metrics.push(Box::new(metric));
    }

    /// Runs evaluation across all datasets and detectors.
    ///
    /// Returns a `DataFrame` with one row per `dataset x detector`, containing:
    /// - `dataset`
    /// - `detector`
    /// - one column per metric
    /// - `time_sec` (elapsed runtime for that detector on that dataset)
    pub fn run(&mut self) -> Result<DataFrame> {
        let entries = loader::list_datasets(&self.data_dir)?;
        if self.metrics.is_empty() {
            self.metrics = all_metrics();
        }
        let metric_names: Vec<String> = self
            .metrics
            .iter()
            .map(|m| m.name().to_string())
            .chain(["time_sec".to_string()])
            .collect();
        let detector_names: Vec<String> =
            self.detector_factories.iter().map(|d| d.name()).collect();
        let mut dataset_col: Vec<String> = Vec::new();
        let mut detector_col: Vec<String> = Vec::new();
        let mut metric_cols: Vec<Vec<f64>> = vec![Vec::new(); self.metrics.len() + 1];

        let total = (entries.len() * self.detector_factories.len()) as u64;
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.cyan} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}  {msg}",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ "),
        );

        for (name, path) in entries {
            let dataset_name = name.clone();
            let dataset = match loader::load_dataset(name, &path) {
                Ok(ds) => ds,
                Err(e) => {
                    pb.println(format!("skipping {}: {e}", path.display()));
                    pb.inc(self.detector_factories.len() as u64);
                    for det_name in &detector_names {
                        dataset_col.push(dataset_name.clone());
                        detector_col.push(det_name.clone());
                        for metric_values in &mut metric_cols {
                            metric_values.push(f64::NAN);
                        }
                    }
                    continue;
                }
            };

            pb.set_message(dataset.name.clone());
            let n_dims = dataset.features.first().map(|f| f.len()).unwrap_or(1);
            let detectors = self
                .detector_factories
                .iter()
                .map(|factory| factory.create(n_dims))
                .collect::<Vec<_>>();
            let ds_results = run_dataset(&dataset, &self.metrics, detectors);
            pb.inc(self.detector_factories.len() as u64);
            for (det_name, det_scores) in detector_names.iter().zip(ds_results.iter()) {
                dataset_col.push(dataset.name.clone());
                detector_col.push(det_name.clone());
                for (mi, value) in det_scores.iter().enumerate() {
                    metric_cols[mi].push(*value);
                }
            }
        }

        pb.finish_and_clear();

        let height = dataset_col.len();
        let mut columns = Vec::with_capacity(2 + metric_names.len());
        columns.push(Column::new("dataset".into(), dataset_col));
        columns.push(Column::new("detector".into(), detector_col));
        for (metric_name, values) in metric_names.iter().zip(metric_cols) {
            columns.push(Column::new(metric_name.as_str().into(), values));
        }

        Ok(DataFrame::new(height, columns)?)
    }
}

/// Runs a single dataset against a set of detector instances.
///
/// Each detector is expected to be freshly initialized for this dataset.
/// Returns metric rows in detector order.
fn run_dataset(
    dataset: &Dataset,
    metrics: &[Box<dyn Metric>],
    mut detectors: Vec<Box<dyn Detector>>,
) -> Vec<Vec<f64>> {
    detectors
        .iter_mut()
        .map(|detector| {
            let start = Instant::now();
            let raw_scores: Vec<f32> = dataset
                .features
                .iter()
                .map(|point| detector.update(point))
                .collect();
            let time_secs = (Instant::now() - start).as_secs_f64();

            let (valid_scores, valid_labels): (Vec<f32>, Vec<u8>) = raw_scores
                .iter()
                .zip(dataset.labels.iter())
                .filter(|(s, _)| !s.is_nan())
                .map(|(&s, &l)| (s, l))
                .unzip();

            if valid_scores.is_empty() {
                return vec![f64::NAN; metrics.len() + 1]; // + 1 (time_secs)
            }

            let norm_scores = minmax_normalize(&valid_scores);
            metrics
                .iter()
                .map(|m| m.score(&valid_labels, &norm_scores))
                .chain([time_secs])
                .collect()
        })
        .collect()
}

/// Command-line arguments shared by every algorithm binary.
#[derive(Parser, Debug)]
pub struct RunArgs {
    /// Directory containing the Touchstone CSV datasets.
    #[arg(long)]
    pub data_dir: PathBuf,
}

/// Parses CLI args, runs a `Touchstone` evaluation for detector `D`, and prints
/// the report. The display name is taken from `D::name()`.
///
/// Used by the [`touchstone_main!`] macro. Call directly if you need to register
/// custom metrics before running.
pub fn run_cli<D>() -> Result<()>
where
    D: Detector + 'static,
{
    let args = RunArgs::parse();
    let mut experiment = Touchstone::new(&args.data_dir);
    experiment.add_detector::<D>();
    let mut report_df = experiment.run()?;

    let mut file = File::create(format!("./touchstone-{}.csv", D::name())).unwrap();
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut report_df)
        .unwrap();

    Ok(())
}

/// Generates a `fn main` that runs the given `Detector`.
///
/// Usage in an algorithm crate's `src/main.rs`:
///
/// ```ignore
/// use touchstone_rs::{Detector, touchstone_main};
///
/// struct MyDetector;
/// impl Detector for MyDetector {
///     fn name() -> &'static str { "MyDetector" }
///     /* ... */
/// }
///
/// touchstone_main!(MyDetector);
/// ```
#[macro_export]
macro_rules! touchstone_main {
    ($detector:ty) => {
        fn main() -> $crate::Result<()> {
            $crate::run_cli::<$detector>()
        }
    };
}
