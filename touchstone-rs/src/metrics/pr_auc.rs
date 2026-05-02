use super::Metric;

/// Precision-Recall Area Under Curve metric.
pub struct PrAuc;

/// Average Precision metric: area under the precision-recall curve.
pub struct AveragePrecision;

impl Metric for PrAuc {
    fn name(&self) -> &str {
        "PR-AUC"
    }

    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        pr_auc(labels, scores)
    }
}

impl Metric for AveragePrecision {
    fn name(&self) -> &str {
        "AvgPrec"
    }

    fn score(&self, labels: &[u8], scores: &[f32]) -> f64 {
        average_precision(labels, scores)
    }
}

/// Pairs each score with its label and sorts descending by score.
fn sorted_pairs(labels: &[u8], scores: &[f32]) -> Vec<(f32, u8)> {
    let mut pairs: Vec<(f32, u8)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    pairs
}

/// Computes the Precision-Recall AUC score using the trapezoidal rule.
pub(crate) fn pr_auc(labels: &[u8], scores: &[f32]) -> f64 {
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    if n_pos == 0 {
        return f64::NAN;
    }

    let pairs = sorted_pairs(labels, scores);
    let n = pairs.len();

    // Walk from highest score down, computing precision/recall at each threshold change.
    let mut auc = 0.0_f64;
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut prev_rec = 0.0_f64;
    let mut prev_prec = 1.0_f64;
    let mut i = 0;

    while i < n {
        let thresh = pairs[i].0;
        while i < n && pairs[i].0 == thresh {
            if pairs[i].1 == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
            i += 1;
        }
        let prec = tp as f64 / (tp + fp) as f64;
        let rec = tp as f64 / n_pos as f64;
        auc += (rec - prev_rec) * (prec + prev_prec) / 2.0;
        prev_rec = rec;
        prev_prec = prec;
    }
    auc
}

/// Computes PR-AUC with dilated anomaly ranges.
#[allow(dead_code)]
pub(crate) fn pr_auc_buffered(labels: &[u8], scores: &[f32], buffer: usize) -> f64 {
    use super::roc_auc::dilate;
    let dilated = dilate(labels, buffer);
    pr_auc(&dilated, scores)
}

/// Computes the Average Precision metric by integrating precision at recall transitions.
fn average_precision(labels: &[u8], scores: &[f32]) -> f64 {
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    if n_pos == 0 {
        return f64::NAN;
    }

    let pairs = sorted_pairs(labels, scores);
    let mut ap = 0.0_f64;
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut prev_rec = 0.0_f64;

    for (_, label) in &pairs {
        if *label == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        let prec = tp as f64 / (tp + fp) as f64;
        let rec = tp as f64 / n_pos as f64;
        ap += (rec - prev_rec) * prec;
        prev_rec = rec;
    }
    ap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_pr_auc() {
        let labels = vec![0, 0, 1, 1];
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let auc = pr_auc(&labels, &scores);
        assert!((auc - 1.0).abs() < 1e-9, "got {auc}");
    }

    #[test]
    fn average_precision_perfect() {
        let labels = vec![0, 0, 1, 1];
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let ap = average_precision(&labels, &scores);
        assert!((ap - 1.0).abs() < 1e-9, "got {ap}");
    }
}
