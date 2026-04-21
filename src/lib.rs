#![doc = include_str!("../README.md")]

pub mod loader;
pub mod metrics;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use loader::Dataset;
use metrics::{Metric, all_metrics, minmax_normalize};
use polars::prelude::{Column, DataFrame};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Streaming detector interface used by Touchstone during evaluation.
///
/// `update` is called once per point and should return an anomaly score.
/// Returning `NaN` is allowed (for example during warmup).
pub trait Detector: Send {
    /// Updates detector state with the next point and returns a score.
    fn update(&mut self, point: &[f32]) -> f32;
}

/// Factory for creating fresh detector instances with a specific dimensionality.
trait DetectorFactory: Send {
    /// Returns the display name of this detector.
    fn name(&self) -> &'static str;
    /// Creates a new detector instance configured for the given dimensionality.
    fn create(&self, n_dims: usize) -> Box<dyn Detector>;
}

/// Erases the concrete detector type and wraps a factory closure for dynamic dispatch.
struct FactoryDetector<D, F> {
    /// Display name used in the output DataFrame.
    name: &'static str,
    /// Constructor called with dataset dimensionality.
    factory: F,
    /// Marker binding this factory to the detector type `D`.
    _detector: PhantomData<D>,
}

impl<D, F> DetectorFactory for FactoryDetector<D, F>
where
    D: Detector + 'static,
    F: Fn(usize) -> D + Send + 'static,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn create(&self, n_dims: usize) -> Box<dyn Detector> {
        Box::new((self.factory)(n_dims))
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

    /// Registers a detector factory.
    ///
    /// The factory receives the dataset dimensionality at runtime and should
    /// return a fresh detector instance for that dimensionality.
    pub fn add_detector<D, F>(&mut self, name: &'static str, factory: F)
    where
        D: Detector + 'static,
        F: Fn(usize) -> D + Send + 'static,
    {
        let detector_factory = FactoryDetector::<D, F> {
            name,
            factory,
            _detector: PhantomData,
        };
        self.detector_factories.push(Box::new(detector_factory));
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
        let detector_names: Vec<&'static str> =
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
                        detector_col.push((*det_name).to_string());
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
                detector_col.push((*det_name).to_string());
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
        for (metric_name, values) in metric_names.iter().zip(metric_cols.into_iter()) {
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
