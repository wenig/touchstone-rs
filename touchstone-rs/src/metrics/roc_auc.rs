use super::Metric;

/// Receiver Operating Characteristic Area Under Curve metric.
pub struct RocAuc;

impl Metric for RocAuc {
    fn name(&self) -> &str {
        "ROC-AUC"
    }

    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        roc_auc(labels, scores)
    }
}

/// Computes the ROC-AUC score using the trapezoidal rule.
pub(crate) fn roc_auc(labels: &[u8], scores: &[f32]) -> f64 {
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    let n_neg = labels.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }

    let mut pairs: Vec<(f32, u8)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Walk the sorted list and trace the ROC curve via trapezoidal rule.
    let mut auc = 0.0_f64;
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut prev_tp = 0_usize;
    let mut prev_fp = 0_usize;
    let mut i = 0;

    while i < pairs.len() {
        // Advance past all ties at the same score.
        let thresh = pairs[i].0;
        while i < pairs.len() && pairs[i].0 == thresh {
            if pairs[i].1 == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
            i += 1;
        }
        let tpr = tp as f64 / n_pos as f64;
        let fpr = fp as f64 / n_neg as f64;
        let prev_tpr = prev_tp as f64 / n_pos as f64;
        let prev_fpr = prev_fp as f64 / n_neg as f64;
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        prev_tp = tp;
        prev_fp = fp;
    }
    auc
}

/// Computes ROC-AUC with dilated anomaly ranges, used in VUS metrics.
///
/// Expands each anomaly range by `buffer` positions on each side.
pub(crate) fn roc_auc_buffered(labels: &[u8], scores: &[f32], buffer: usize) -> f64 {
    let dilated: Vec<u8> = dilate(labels, buffer);
    roc_auc(&dilated, scores)
}

/// Dilates binary labels by expanding 1s by `buffer` positions in each direction.
pub(crate) fn dilate(labels: &[u8], buffer: usize) -> Vec<u8> {
    if buffer == 0 {
        return labels.to_vec();
    }
    let n = labels.len();
    let mut out = vec![0u8; n];
    for (i, &l) in labels.iter().enumerate() {
        if l == 1 {
            let lo = i.saturating_sub(buffer);
            let hi = (i + buffer + 1).min(n);
            for o in &mut out[lo..hi] {
                *o = 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_classifier() {
        let labels = vec![0, 0, 0, 1, 1];
        let scores = vec![0.1, 0.2, 0.3, 0.8, 0.9];
        let auc = roc_auc(&labels, &scores);
        assert!((auc - 1.0).abs() < 1e-9, "got {auc}");
    }

    #[test]
    fn inverse_classifier() {
        let labels = vec![0, 0, 0, 1, 1];
        let scores = vec![0.9, 0.8, 0.7, 0.2, 0.1];
        let auc = roc_auc(&labels, &scores);
        assert!((auc - 0.0).abs() < 1e-9, "got {auc}");
    }

    #[test]
    fn random_classifier_is_near_half() {
        // Alternating labels with non-informative score.
        let labels: Vec<u8> = (0..100).map(|i| i % 2).collect();
        let scores: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let auc = roc_auc(&labels, &scores);
        assert!((auc - 0.5).abs() < 0.1, "got {auc}");
    }
}
