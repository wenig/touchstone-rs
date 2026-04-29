use super::*;

fn sine_point(t: usize) -> Vec<f32> {
    vec![(t as f32 * 0.1).sin()]
}

fn warm_up(det: &mut KMeans) {
    for i in 0..WARMUP_LEN {
        det.update(&sine_point(i));
    }
}

// ── extract_windows ───────────────────────────────────────────────────────────

#[test]
fn extract_windows_count_and_boundaries() {
    let data: Vec<Vec<f32>> = (0..W + 5).map(|i| vec![i as f32]).collect();
    let windows = KMeans::extract_windows(&data);
    assert_eq!(windows.len(), 6, "expected n - W + 1 windows");
    assert_eq!(windows[0][0], vec![0.0f32]);
    assert_eq!(windows[0][W - 1], vec![(W - 1) as f32]);
    assert_eq!(windows[5][0], vec![5.0f32]);
}

#[test]
fn extract_windows_empty_when_too_short() {
    let data: Vec<Vec<f32>> = (0..W - 1).map(|i| vec![i as f32]).collect();
    assert!(KMeans::extract_windows(&data).is_empty());
}

// ── sample_weighted ───────────────────────────────────────────────────────────

#[test]
fn sample_weighted_zero_total_returns_valid_index() {
    let weights = vec![0.0f32; 5];
    let mut rng = rand::rng();
    let idx = KMeans::sample_weighted(&weights, &mut rng);
    assert!(idx < 5);
}

#[test]
fn sample_weighted_single_dominant_weight() {
    let mut weights = vec![0.0f32; 10];
    weights[7] = 1.0;
    let mut rng = rand::rng();
    for _ in 0..20 {
        assert_eq!(KMeans::sample_weighted(&weights, &mut rng), 7);
    }
}

// ── warmup / initialization ───────────────────────────────────────────────────

#[test]
fn warmup_returns_nan() {
    let mut det = KMeans::new(1);
    for i in 0..WARMUP_LEN {
        let score = det.update(&sine_point(i));
        assert!(
            score.is_nan(),
            "step {i}: expected NaN during warmup, got {score}"
        );
    }
}

#[test]
fn post_warmup_score_is_finite() {
    let mut det = KMeans::new(1);
    warm_up(&mut det);
    let score = det.update(&sine_point(WARMUP_LEN));
    assert!(
        score.is_finite(),
        "expected finite score after warmup, got {score}"
    );
}

#[test]
fn k_centroids_initialized() {
    let mut det = KMeans::new(1);
    warm_up(&mut det);
    assert_eq!(
        det.centroids.len(),
        K,
        "should have K={K} centroids after warmup"
    );
}

#[test]
fn each_centroid_has_w_points() {
    let mut det = KMeans::new(1);
    warm_up(&mut det);
    for (k, c) in det.centroids.iter().enumerate() {
        assert_eq!(c.len(), W, "centroid {k} has wrong window length");
    }
}

#[test]
fn warmup_buffer_freed_after_init() {
    let mut det = KMeans::new(1);
    warm_up(&mut det);
    assert!(
        det.warmup_buffer.is_empty(),
        "warmup_buffer should be freed after init"
    );
}

// ── scoring ───────────────────────────────────────────────────────────────────

#[test]
fn nan_input_returns_nan_and_does_not_poison() {
    let mut det = KMeans::new(1);
    for i in 0..WARMUP_LEN + W {
        det.update(&sine_point(i));
    }
    let nan_score = det.update(&[f32::NAN]);
    assert!(nan_score.is_nan(), "NaN input should return NaN");
    // subsequent normal input should still return a finite score
    let next = det.update(&sine_point(0));
    assert!(
        next.is_finite(),
        "model should be unaffected by NaN input, got {next}"
    );
}

#[test]
fn anomaly_scores_higher_than_normal() {
    let mut det = KMeans::new(1);
    for i in 0..WARMUP_LEN + W {
        det.update(&sine_point(i));
    }

    let normal_scores: Vec<f32> = (0..200)
        .filter_map(|i| {
            let s = det.update(&sine_point(i));
            s.is_finite().then_some(s)
        })
        .collect();

    let anomaly_scores: Vec<f32> = (0..10)
        .filter_map(|_| {
            let s = det.update(&[1000.0f32]);
            s.is_finite().then_some(s)
        })
        .collect();

    assert!(!normal_scores.is_empty(), "no normal scores collected");
    assert!(!anomaly_scores.is_empty(), "no anomaly scores collected");

    let normal_mean = normal_scores.iter().sum::<f32>() / normal_scores.len() as f32;
    let anomaly_mean = anomaly_scores.iter().sum::<f32>() / anomaly_scores.len() as f32;

    assert!(
        anomaly_mean > normal_mean,
        "anomaly mean {anomaly_mean:.4} should exceed normal mean {normal_mean:.4}"
    );
}

#[test]
fn centroid_moves_toward_window() {
    let mut det = KMeans::new(1);
    warm_up(&mut det);
    // fill buffer with a flat 0.5 signal
    for _ in 0..W {
        det.update(&[0.5f32]);
    }
    let window: MultivariateWindow = det.buffer.iter().cloned().collect();
    let (_, dist_before) = det.find_closest(&window);
    // one more 0.5 → buffer unchanged, centroid moves closer
    det.update(&[0.5f32]);
    let (_, dist_after) = det.find_closest(&window);
    assert!(
        dist_after <= dist_before,
        "centroid should move toward window: before={dist_before:.6}, after={dist_after:.6}"
    );
}

// ── misc ──────────────────────────────────────────────────────────────────────

#[test]
fn multivariate_input_scores_finite() {
    let mut det = KMeans::new(3);
    for i in 0..WARMUP_LEN + 1 {
        let t = i as f32;
        det.update(&[(t * 0.1).sin(), (t * 0.2).cos(), t * 0.001]);
    }
    let score = det.update(&[0.5, 0.3, 0.1]);
    assert!(score.is_finite(), "multivariate score should be finite");
}

#[test]
fn name_is_kmeans() {
    assert_eq!(KMeans::name(), "KMeans");
}
