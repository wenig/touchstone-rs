/// Strategy for computing a decision threshold from anomaly scores.
pub trait Threshold: Send + Sync {
    /// Computes the threshold value from the given scores.
    fn threshold(&self, scores: &[f32]) -> f32;
    /// Returns the name of this threshold strategy.
    #[allow(dead_code)]
    fn name(&self) -> &str;
}

/// Fixed score threshold.
#[allow(dead_code)]
pub struct FixedValueThreshold(pub f32);

/// Threshold at the p-th percentile of scores (p in 0–100).
pub struct PercentileThreshold(pub f64);

/// Threshold at mean + k * std.
#[allow(dead_code)]
pub struct SigmaThreshold(pub f64);

impl Threshold for FixedValueThreshold {
    fn threshold(&self, _scores: &[f32]) -> f32 {
        self.0
    }
    fn name(&self) -> &str {
        "fixed"
    }
}

impl Threshold for PercentileThreshold {
    fn threshold(&self, scores: &[f32]) -> f32 {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let idx = ((self.0 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }
    fn name(&self) -> &str {
        "percentile"
    }
}

impl Threshold for SigmaThreshold {
    fn threshold(&self, scores: &[f32]) -> f32 {
        let n = scores.len() as f64;
        let mean = scores.iter().map(|&s| s as f64).sum::<f64>() / n;
        let var = scores
            .iter()
            .map(|&s| (s as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        (mean + self.0 * var.sqrt()) as f32
    }
    fn name(&self) -> &str {
        "sigma"
    }
}

/// Binarizes scores using a threshold: 1 if score >= threshold, 0 otherwise.
pub(crate) fn apply_threshold(scores: &[f32], thresh: f32) -> Vec<u8> {
    scores
        .iter()
        .map(|&s| if s >= thresh { 1 } else { 0 })
        .collect()
}
