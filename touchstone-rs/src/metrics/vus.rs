/// Volume Under Surface metrics (Paparrizos et al., PVLDB 2022).
///
/// Average the range-PR-AUC (or ROC-AUC with dilation) over buffer sizes
/// 0 through `max_buffer`, producing a single robustness-aware score.
use rayon::prelude::*;

use super::{
    Metric,
    range::{Bias, Cardinality, range_pr_auc_impl},
    roc_auc::roc_auc_buffered,
};

/// Volume Under Surface metric for range-based PR curve (Paparrizos et al., PVLDB 2022).
///
/// Averages range-PR-AUC over buffer sizes 0 through `max_buffer`.
pub struct RangePrVus {
    /// Largest dilation/buffer size included in the VUS average.
    pub max_buffer: usize,
    /// Penalization policy when multiple predicted ranges match one true range.
    pub cardinality: Cardinality,
    /// Positional weighting within each range.
    pub bias: Bias,
    /// Maximum number of thresholds sampled for each buffered PR-AUC.
    pub max_samples: usize,
}

/// Volume Under Surface metric for range-based ROC curve (Paparrizos et al., PVLDB 2022).
///
/// Averages range-ROC-AUC over dilated anomaly ranges (buffers 0 through `max_buffer`).
pub struct RangeRocVus {
    /// Largest dilation/buffer size included in the VUS average.
    pub max_buffer: usize,
}

impl Default for RangePrVus {
    fn default() -> Self {
        Self {
            max_buffer: 200,
            cardinality: Cardinality::One,
            bias: Bias::Flat,
            max_samples: 20,
        }
    }
}

impl Default for RangeRocVus {
    fn default() -> Self {
        Self { max_buffer: 200 }
    }
}

impl Metric for RangePrVus {
    fn name(&self) -> &str {
        "RangePR-VUS"
    }

    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        let cardinality = self.cardinality;
        let bias = self.bias;
        let max_samples = self.max_samples;
        let values: Vec<f64> = (0..=self.max_buffer)
            .into_par_iter()
            .map(|buf| {
                let dilated = super::roc_auc::dilate(labels, buf);
                range_pr_auc_impl(&dilated, scores, cardinality, bias, max_samples)
            })
            .filter(|v| v.is_finite())
            .collect();
        if values.is_empty() {
            return f64::NAN;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }
}

impl Metric for RangeRocVus {
    fn name(&self) -> &str {
        "RangeROC-VUS"
    }

    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        let values: Vec<f64> = (0..=self.max_buffer)
            .into_par_iter()
            .map(|buf| roc_auc_buffered(labels, scores, buf))
            .filter(|v| v.is_finite())
            .collect();
        if values.is_empty() {
            return f64::NAN;
        }
        values.iter().sum::<f64>() / values.len() as f64
    }
}
