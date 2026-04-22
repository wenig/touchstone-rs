#![allow(unused_imports)]

mod classification;
mod pr_auc;
mod range;
mod roc_auc;
mod thresholding;
mod vus;

pub use classification::{F1Score, Precision, Recall};
pub use pr_auc::{AveragePrecision, PrAuc};
pub use range::{RangeAuc, RangeFScore, RangePrecision, RangeRecall};
pub use roc_auc::RocAuc;
pub use thresholding::{FixedValueThreshold, PercentileThreshold, SigmaThreshold, Threshold};
pub use vus::{RangePrVus, RangeRocVus};

pub trait Metric: Send + Sync {
    /// Returns the name of this metric.
    fn name(&self) -> &str;
    /// Computes the metric score.
    ///
    /// # Arguments
    /// * `labels` - Ground-truth (0 = normal, 1 = anomaly)
    /// * `scores` - Anomaly score (higher = more anomalous); already NaN-filtered and minmax-normalized to [0,1]
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64;
}

/// Returns the default set of evaluation metrics.
///
/// Includes ROC-AUC, PR-AUC, Average Precision, classification metrics at 90th percentile,
/// and range-based/VUS metrics for assessing range-wise detection quality.
pub fn all_metrics() -> Vec<Box<dyn Metric>> {
    vec![
        Box::new(RocAuc),
        Box::new(PrAuc),
        Box::new(AveragePrecision),
        Box::new(Precision::new(PercentileThreshold(90.0))),
        Box::new(Recall::new(PercentileThreshold(90.0))),
        Box::new(F1Score::new(PercentileThreshold(90.0))),
        Box::new(RangePrecision::default()),
        Box::new(RangeRecall::default()),
        Box::new(RangeFScore::default()),
        Box::new(RangeAuc::default()),
        Box::new(RangePrVus::default()),
        Box::new(RangeRocVus::default()),
    ]
}

/// Normalizes scores to the range [0, 1] using min-max scaling.
///
/// If all scores are identical (range < epsilon), returns a vector of zeros.
pub fn minmax_normalize(scores: &[f32]) -> Vec<f32> {
    let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range < f32::EPSILON {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|&s| (s - min) / range).collect()
}
