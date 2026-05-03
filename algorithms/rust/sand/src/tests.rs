use std::f32::consts::PI;

use crate::distance::{msbd, ncc_c, znorm};

// ── helpers ───────────────────────────────────────────────────────────────────

fn sine(n: usize, freq: f32, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|t| (2.0 * PI * freq * t as f32 / n as f32 + phase).sin())
        .collect()
}

/// Build a `(n, d)` row-major multivariate signal; each channel is a
/// sine at a different frequency to ensure the channels are independent.
fn mv_signal(n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    for ch in 0..d {
        let sig = sine(n, 2.0 + ch as f32, ch as f32 * 0.5);
        for t in 0..n {
            out[t * d + ch] = sig[t];
        }
    }
    out
}

// ── ncc_c ─────────────────────────────────────────────────────────────────────

#[test]
fn ncc_c_self_correlation_is_one_at_lag_zero() {
    let x = znorm(&sine(32, 3.0, 0.0));
    let ncc = ncc_c(&x, &x);
    let n = x.len();
    // Lag 0 is stored at index n-1 in the 2n-1 output.
    assert!(
        (ncc[n - 1] - 1.0).abs() < 1e-5,
        "NCC at lag 0 should be 1.0, got {}",
        ncc[n - 1]
    );
}

#[test]
fn ncc_c_values_are_bounded_in_minus_one_to_one() {
    let x = znorm(&sine(32, 3.0, 0.0));
    let y = znorm(&sine(32, 5.0, 1.2));
    for &v in ncc_c(&x, &y).iter() {
        assert!(
            v >= -1.0 - 1e-5 && v <= 1.0 + 1e-5,
            "ncc value {v} is outside [-1, 1]"
        );
    }
}

#[test]
fn ncc_c_detects_shift() {
    // y is x delayed by `shift` positions (zero-padded on the left).
    // R_w(x, y) peaks at w = -shift, stored at index (n-1) - shift.
    let n = 64;
    let shift: usize = 5;
    let x = znorm(&sine(n, 4.0, 0.0));
    let mut y_raw = vec![0.0f32; n];
    for t in shift..n {
        y_raw[t] = x[t - shift];
    }
    let y = znorm(&y_raw);
    let ncc = ncc_c(&x, &y);
    let best_idx = ncc
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();
    let best_lag = best_idx as isize - (n as isize - 1);
    assert_eq!(
        best_lag,
        -(shift as isize),
        "expected lag -{shift}, got {best_lag}"
    );
}

// ── msbd ──────────────────────────────────────────────────────────────────────

#[test]
fn msbd_self_distance_is_zero() {
    let n = 30;
    let d = 4;
    let x = mv_signal(n, d);
    let dist = msbd(&x, &x, n, d);
    assert!(dist < 1e-5, "self-distance should be 0, got {dist}");
}

#[test]
fn msbd_output_is_in_valid_range() {
    let n = 24;
    let d = 3;
    let x = mv_signal(n, d);
    let y = mv_signal(n, d);
    let dist = msbd(&x, &y, n, d);
    assert!(
        dist >= 0.0 && dist <= 2.0 + 1e-5,
        "msbd {dist} is outside [0, 2]"
    );
}

#[test]
fn msbd_shifted_copy_is_closer_than_orthogonal_signal() {
    let n = 48;
    let d = 3;
    let x = mv_signal(n, d);

    // Build a time-shifted copy of x (delayed by 4 steps, zero-padded).
    let shift = 4;
    let mut x_shifted = vec![0.0f32; n * d];
    for t in shift..n {
        for ch in 0..d {
            x_shifted[t * d + ch] = x[(t - shift) * d + ch];
        }
    }

    // Build an orthogonal-ish signal at a different frequency.
    let x_ortho = mv_signal(n, d)
        .iter()
        .enumerate()
        .map(|(i, &v)| v + (i as f32 * 1.7).cos())
        .collect::<Vec<_>>();

    let dist_shifted = msbd(&x, &x_shifted, n, d);
    let dist_ortho = msbd(&x, &x_ortho, n, d);

    assert!(
        dist_shifted < dist_ortho,
        "shifted ({dist_shifted:.4}) should be closer than orthogonal ({dist_ortho:.4})"
    );
}

#[test]
fn msbd_d1_matches_manual_ncc_computation() {
    // For d=1, msbd should equal 1 - max(ncc_c(znorm(x), znorm(y))).
    let n = 20;
    let x_raw = sine(n, 3.0, 0.0);
    let y_raw = sine(n, 3.0, 0.8);

    let dist_msbd = msbd(&x_raw, &y_raw, n, 1);

    let xn = znorm(&x_raw);
    let yn = znorm(&y_raw);
    let ncc = ncc_c(&xn, &yn);
    let max_ncc = ncc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let dist_manual = (1.0 - max_ncc).max(0.0);

    assert!(
        (dist_msbd - dist_manual).abs() < 1e-5,
        "d=1 msbd ({dist_msbd:.6}) should match manual ncc ({dist_manual:.6})"
    );
}
