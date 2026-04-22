use super::{
    Metric,
    thresholding::{Threshold, apply_threshold},
};

/// Precision metric: ratio of true positives to predicted positives.
pub struct Precision {
    /// Threshold strategy used to binarize anomaly scores.
    threshold: Box<dyn Threshold>,
}

/// Recall metric: ratio of true positives to actual positives.
pub struct Recall {
    /// Threshold strategy used to binarize anomaly scores.
    threshold: Box<dyn Threshold>,
}

/// F1 Score metric: harmonic mean of precision and recall.
pub struct F1Score {
    /// Threshold strategy used to binarize anomaly scores.
    threshold: Box<dyn Threshold>,
}

impl Precision {
    /// Creates a new Precision metric with the given threshold strategy.
    pub fn new(t: impl Threshold + 'static) -> Self {
        Self {
            threshold: Box::new(t),
        }
    }
}

impl Recall {
    /// Creates a new Recall metric with the given threshold strategy.
    pub fn new(t: impl Threshold + 'static) -> Self {
        Self {
            threshold: Box::new(t),
        }
    }
}

impl F1Score {
    /// Creates a new F1 Score metric with the given threshold strategy.
    pub fn new(t: impl Threshold + 'static) -> Self {
        Self {
            threshold: Box::new(t),
        }
    }
}

/// Computes confusion matrix counts: (true_positives, false_positives, false_negatives).
fn confusion(labels: &[u8], preds: &[u8]) -> (usize, usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_ = 0;
    for (&l, &p) in labels.iter().zip(preds.iter()) {
        match (l, p) {
            (1, 1) => tp += 1,
            (0, 1) => fp += 1,
            (1, 0) => fn_ += 1,
            _ => {}
        }
    }
    (tp, fp, fn_)
}

impl Metric for Precision {
    fn name(&self) -> &str {
        "Precision"
    }
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        let thresh = self.threshold.threshold(scores);
        let preds = apply_threshold(scores, thresh);
        let (tp, fp, _) = confusion(labels, &preds);
        if tp + fp == 0 {
            return 0.0;
        }
        tp as f64 / (tp + fp) as f64
    }
}

impl Metric for Recall {
    fn name(&self) -> &str {
        "Recall"
    }
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        let thresh = self.threshold.threshold(scores);
        let preds = apply_threshold(scores, thresh);
        let (tp, _, fn_) = confusion(labels, &preds);
        if tp + fn_ == 0 {
            return f64::NAN;
        }
        tp as f64 / (tp + fn_) as f64
    }
}

impl Metric for F1Score {
    fn name(&self) -> &str {
        "F1"
    }
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        let thresh = self.threshold.threshold(scores);
        let preds = apply_threshold(scores, thresh);
        let (tp, fp, fn_) = confusion(labels, &preds);
        let denom = 2 * tp + fp + fn_;
        if denom == 0 {
            return 0.0;
        }
        2.0 * tp as f64 / denom as f64
    }
}
