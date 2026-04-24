/// Range-based precision and recall from Tatbul et al., NeurIPS 2018.
/// "Precision and Recall for Time Series"
use super::{Metric, thresholding::apply_threshold};

/// Penalization policy when multiple predicted ranges match one true range.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Cardinality {
    /// Each predicted range contributes fully (1.0) regardless of multiplicity.
    One,
    /// Multiple overlaps incur a 1/count penalty.
    Reciprocal,
}

/// Positional weighting within each range.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Bias {
    /// All positions within a range have equal weight.
    Flat,
    /// Weight decreases from start to end (front-loading).
    Front,
    /// Weight is highest at the center, declining toward edges.
    Middle,
    /// Weight increases from start to end (back-loading).
    Back,
}

/// Extracts contiguous ranges of 1s from a binary vector, returning start/end indices.
fn extract_ranges(binary: &[u8]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = None;
    for (i, &v) in binary.iter().enumerate() {
        match (v, start) {
            (1, None) => start = Some(i),
            (0, Some(s)) => {
                ranges.push((s, i - 1));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        ranges.push((s, binary.len() - 1));
    }
    ranges
}

/// Computes the positional weight for a position within a range given a bias strategy.
fn delta(pos: usize, range_start: usize, range_end: usize, bias: Bias) -> f64 {
    let len = (range_end - range_start + 1) as f64;
    match bias {
        Bias::Flat => 1.0,
        Bias::Front => {
            let i = (pos - range_start + 1) as f64;
            (2.0 * (len - i + 1.0)) / (len * (len + 1.0))
        }
        Bias::Back => {
            let i = (pos - range_start + 1) as f64;
            (2.0 * i) / (len * (len + 1.0))
        }
        Bias::Middle => {
            let i = (pos - range_start + 1) as f64;
            let mid = (len + 1.0) / 2.0;
            let dist = (i - mid).abs();
            let peak = if len % 2.0 == 0.0 { 0.5 } else { 1.0 };
            // Triangle: 0 at edges, peak at center
            if len == 1.0 {
                1.0
            } else {
                peak - dist * peak / (len / 2.0).ceil()
            }
        }
    }
}

/// Computes the weighted overlap between a predicted and real range.
fn omega(pred: (usize, usize), real: (usize, usize), bias: Bias) -> f64 {
    let overlap_start = pred.0.max(real.0);
    let overlap_end = pred.1.min(real.1);
    if overlap_start > overlap_end {
        return 0.0;
    }
    let my_len = (pred.1 - pred.0 + 1) as f64;
    let weighted_overlap: f64 = (overlap_start..=overlap_end)
        .map(|p| delta(p, pred.0, pred.1, bias))
        .sum();
    let total_weight: f64 = (pred.0..=pred.1)
        .map(|p| delta(p, pred.0, pred.1, bias))
        .sum();
    if total_weight < 1e-12 {
        return 0.0;
    }
    weighted_overlap / total_weight * (overlap_end - overlap_start + 1) as f64 / my_len
}

/// Computes the cardinality penalty applied to overlap counts.
fn gamma(overlap_count: usize, cardinality: Cardinality) -> f64 {
    match cardinality {
        Cardinality::One => 1.0,
        Cardinality::Reciprocal => {
            if overlap_count == 0 {
                0.0
            } else {
                1.0 / overlap_count as f64
            }
        }
    }
}

/// Score a single anomaly range (either predicted or real) against the set of
/// reference ranges on the other side.
///
/// When scoring precision: `my_range` = predicted range, `ref_ranges` = real ranges.
/// When scoring recall:    `my_range` = real range,      `ref_ranges` = predicted ranges.
fn range_score(
    my_range: (usize, usize),
    ref_ranges: &[(usize, usize)],
    alpha: f64,
    cardinality: Cardinality,
    bias: Bias,
) -> f64 {
    let mut overlap_reward = 0.0;
    let mut overlap_count = 0;

    for &r in ref_ranges {
        let ov = omega(my_range, r, bias);
        if ov > 0.0 {
            overlap_reward += ov;
            overlap_count += 1;
        }
    }

    let existence = if overlap_count > 0 { 1.0 } else { 0.0 };
    overlap_reward *= gamma(overlap_count, cardinality);

    alpha * existence + (1.0 - alpha) * overlap_reward
}

/// Computes range-based precision: average score of predicted ranges against real ranges.
pub(crate) fn range_precision_raw(
    real: &[u8],
    pred: &[u8],
    alpha: f64,
    cardinality: Cardinality,
    bias: Bias,
) -> f64 {
    let pred_ranges = extract_ranges(pred);
    if pred_ranges.is_empty() {
        return 0.0;
    }
    let real_ranges = extract_ranges(real);
    let sum: f64 = pred_ranges
        .iter()
        .map(|&p| range_score(p, &real_ranges, alpha, cardinality, bias))
        .sum();
    sum / pred_ranges.len() as f64
}

/// Computes range-based recall: average score of real ranges against predicted ranges.
pub(crate) fn range_recall_raw(
    real: &[u8],
    pred: &[u8],
    alpha: f64,
    cardinality: Cardinality,
    bias: Bias,
) -> f64 {
    let real_ranges = extract_ranges(real);
    if real_ranges.is_empty() {
        return f64::NAN;
    }
    let pred_ranges = extract_ranges(pred);
    let sum: f64 = real_ranges
        .iter()
        .map(|&r| range_score(r, &pred_ranges, alpha, cardinality, bias))
        .sum();
    sum / real_ranges.len() as f64
}

/// Computes the F-score from range precision and recall with a given beta weight.
fn range_fscore(prec: f64, rec: f64, beta: f64) -> f64 {
    let denom = beta * beta * prec + rec;
    if denom < 1e-12 {
        return 0.0;
    }
    (1.0 + beta * beta) * prec * rec / denom
}

// ─── public metric structs ──────────────────────────────────────────────────

/// Range-based precision metric (Tatbul et al., NeurIPS 2018).
pub struct RangePrecision {
    /// Weight between overlap-only (0.0) and existence-aware (1.0) scoring.
    pub alpha: f64,
    /// Penalization policy when multiple predicted ranges match one true range.
    pub cardinality: Cardinality,
    /// Positional weighting within each range.
    pub bias: Bias,
    /// Score percentile used to derive a binary prediction threshold.
    pub percentile: f64,
}

/// Range-based recall metric (Tatbul et al., NeurIPS 2018).
pub struct RangeRecall {
    /// Weight between overlap-only (0.0) and existence-aware (1.0) scoring.
    pub alpha: f64,
    /// Penalization policy when multiple predicted ranges match one true range.
    pub cardinality: Cardinality,
    /// Positional weighting within each range.
    pub bias: Bias,
    /// Score percentile used to derive a binary prediction threshold.
    pub percentile: f64,
}

/// Range-based F-score metric (Tatbul et al., NeurIPS 2018).
pub struct RangeFScore {
    /// Relative recall weight in F-score (`1.0` gives F1).
    pub beta: f64,
    /// Alpha used in the precision component.
    pub p_alpha: f64,
    /// Alpha used in the recall component.
    pub r_alpha: f64,
    /// Penalization policy for range multiplicity.
    pub cardinality: Cardinality,
    /// Bias used in the precision component.
    pub p_bias: Bias,
    /// Bias used in the recall component.
    pub r_bias: Bias,
    /// Score percentile used to derive a binary prediction threshold.
    pub percentile: f64,
}

/// AUC of the range-based PR curve, sweeping thresholds.
pub struct RangeAuc {
    /// Penalization policy when multiple predicted ranges match one true range.
    pub cardinality: Cardinality,
    /// Positional weighting within each range.
    pub bias: Bias,
    /// Maximum number of thresholds sampled when approximating the curve.
    pub max_samples: usize,
}

impl Default for RangePrecision {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            cardinality: Cardinality::One,
            bias: Bias::Flat,
            percentile: 90.0,
        }
    }
}

impl Default for RangeRecall {
    fn default() -> Self {
        Self {
            alpha: 0.0,
            cardinality: Cardinality::One,
            bias: Bias::Flat,
            percentile: 90.0,
        }
    }
}

impl Default for RangeFScore {
    fn default() -> Self {
        Self {
            beta: 1.0,
            p_alpha: 0.0,
            r_alpha: 0.0,
            cardinality: Cardinality::One,
            p_bias: Bias::Flat,
            r_bias: Bias::Flat,
            percentile: 90.0,
        }
    }
}

impl Default for RangeAuc {
    fn default() -> Self {
        Self {
            cardinality: Cardinality::One,
            bias: Bias::Flat,
            max_samples: 50,
        }
    }
}

impl Metric for RangePrecision {
    fn name(&self) -> &str {
        "RangePrec"
    }
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((self.percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        let thresh = sorted[idx.min(sorted.len() - 1)];
        let pred = apply_threshold(scores, thresh);
        range_precision_raw(labels, &pred, self.alpha, self.cardinality, self.bias)
    }
}

impl Metric for RangeRecall {
    fn name(&self) -> &str {
        "RangeRec"
    }
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((self.percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        let thresh = sorted[idx.min(sorted.len() - 1)];
        let pred = apply_threshold(scores, thresh);
        range_recall_raw(labels, &pred, self.alpha, self.cardinality, self.bias)
    }
}

impl Metric for RangeFScore {
    fn name(&self) -> &str {
        "RangeF1"
    }
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((self.percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        let thresh = sorted[idx.min(sorted.len() - 1)];
        let pred = apply_threshold(scores, thresh);
        let p = range_precision_raw(labels, &pred, self.p_alpha, self.cardinality, self.p_bias);
        let r = range_recall_raw(labels, &pred, self.r_alpha, self.cardinality, self.r_bias);
        range_fscore(p, r, self.beta)
    }
}

/// Computes the area under the range-based precision-recall curve.
pub(crate) fn range_pr_auc_impl(
    labels: &[u8],
    scores: &[f32],
    cardinality: Cardinality,
    bias: Bias,
    max_samples: usize,
) -> f64 {
    // Collect unique thresholds (capped at max_samples evenly spaced).
    let mut sorted_scores = scores.to_vec();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_scores.dedup_by(|a, b| (*a - *b).abs() < f32::EPSILON);

    let step = if sorted_scores.len() <= max_samples {
        1
    } else {
        sorted_scores.len() / max_samples
    };

    let thresholds: Vec<f32> = sorted_scores.into_iter().step_by(step.max(1)).collect();

    let mut points: Vec<(f64, f64)> = thresholds
        .iter()
        .map(|&t| {
            let pred = apply_threshold(scores, t);
            let p = range_precision_raw(labels, &pred, 0.0, cardinality, bias);
            let r = range_recall_raw(labels, &pred, 0.0, cardinality, bias);
            (r, p)
        })
        .collect();

    // Add sentinel endpoints.
    points.push((0.0, 1.0));
    points.push((1.0, 0.0));
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-12);

    // Trapezoidal integration.
    let mut auc = 0.0;
    for w in points.windows(2) {
        let (r0, p0) = w[0];
        let (r1, p1) = w[1];
        auc += (r1 - r0) * (p0 + p1) / 2.0;
    }
    auc
}

impl Metric for RangeAuc {
    fn name(&self) -> &str {
        "RangePR-AUC"
    }
    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        range_pr_auc_impl(
            labels,
            scores,
            self.cardinality,
            self.bias,
            self.max_samples,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_ranges_basic() {
        let b = vec![0, 1, 1, 0, 1, 0];
        assert_eq!(extract_ranges(&b), vec![(1, 2), (4, 4)]);
    }

    #[test]
    fn omega_full_overlap() {
        // Identical ranges → overlap score = 1.0
        let score = omega((2, 5), (2, 5), Bias::Flat);
        assert!((score - 1.0).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn omega_no_overlap() {
        let score = omega((0, 2), (5, 8), Bias::Flat);
        assert!((score).abs() < 1e-9, "got {score}");
    }

    #[test]
    fn gamma_reciprocal_penalizes() {
        assert!((gamma(1, Cardinality::One) - 1.0).abs() < 1e-9);
        assert!((gamma(2, Cardinality::Reciprocal) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn range_precision_perfect() {
        let real = vec![0, 0, 1, 1, 1, 0, 0];
        let pred = vec![0, 0, 1, 1, 1, 0, 0];
        let p = range_precision_raw(&real, &pred, 0.0, Cardinality::One, Bias::Flat);
        assert!((p - 1.0).abs() < 1e-9, "got {p}");
    }

    #[test]
    fn range_recall_perfect() {
        let real = vec![0, 0, 1, 1, 1, 0, 0];
        let pred = vec![0, 0, 1, 1, 1, 0, 0];
        let r = range_recall_raw(&real, &pred, 0.0, Cardinality::One, Bias::Flat);
        assert!((r - 1.0).abs() < 1e-9, "got {r}");
    }
}
