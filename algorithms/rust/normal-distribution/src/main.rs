//! Sliding-window anomaly detector based on per-dimension z-scores.
//!
//! Each incoming point is scored as the Euclidean norm of its per-dimension
//! z-scores computed against the current window of recent observations.

use std::collections::VecDeque;

use touchstone_rs::{Detector, touchstone_main};

/// Default number of observations retained in the sliding window.
const CAPACITY: usize = 100;

/// Anomaly detector that scores points using a sliding-window normal distribution.
///
/// Maintains a fixed-size window of recent observations and scores each new
/// point as the L2 norm of its per-dimension z-scores against that window.
/// Running sums (`sum`, `sum_sq`) allow O(1) mean and variance updates.
pub struct NormalDistribution {
    buffer: VecDeque<Vec<f32>>,
    capacity: usize,
    dim: usize,
    sum: Vec<f32>,
    sum_sq: Vec<f32>,
}

impl NormalDistribution {
    /// Computes the anomaly score for `x` against the current window.
    ///
    /// Returns the L2 norm of per-dimension z-scores: `sqrt(Σ((xᵢ - μᵢ) / σᵢ)²)`.
    /// Dimensions with zero variance contribute 0. Returns `NaN` when the window
    /// holds fewer than 2 points (insufficient to estimate variance).
    fn zscore_against_current(&self, x: &[f32]) -> f32 {
        let n = self.buffer.len();
        if n < 2 {
            return f32::NAN;
        }

        (0..self.dim)
            .map(|i| {
                let mean = self.sum[i] / n as f32;
                let var = (self.sum_sq[i] - n as f32 * mean * mean) / (n as f32 - 1.0);

                let std = var.max(0.0).sqrt();

                if std == 0.0 { 0.0 } else { (x[i] - mean) / std }
            })
            .map(|d| d * d)
            .sum::<f32>()
            .sqrt()
    }
}

impl Detector for NormalDistribution {
    fn name() -> &'static str {
        "NormalDistribution"
    }

    /// Creates a new detector for `n_dimensions`-dimensional input with a window of [`CAPACITY`].
    fn new(n_dimensions: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(CAPACITY),
            capacity: CAPACITY,
            dim: n_dimensions,
            sum: vec![0.0; n_dimensions],
            sum_sq: vec![0.0; n_dimensions],
        }
    }

    /// Scores `point` against the current window, then adds it to the window.
    ///
    /// If the window is full the oldest observation is evicted first. Returns
    /// the anomaly score computed *before* the point is ingested, so the point
    /// is not compared against itself.
    fn update(&mut self, point: &[f32]) -> f32 {
        let zscore = self.zscore_against_current(point);

        if self.buffer.len() == self.capacity
            && let Some(old) = self.buffer.pop_front()
        {
            for (i, &val) in old.iter().enumerate().take(self.dim) {
                self.sum[i] -= val;
                self.sum_sq[i] -= val.powi(2);
            }
        }

        for (i, &p) in point.iter().enumerate().take(self.dim) {
            self.sum[i] += p;
            self.sum_sq[i] += p.powi(2);
        }

        self.buffer.push_back(point.to_vec());

        zscore
    }
}

touchstone_main!(NormalDistribution);
