use std::collections::VecDeque;

use rand::{Rng, RngExt};
use touchstone_rs::{Detector, touchstone_main};

const K: usize = 10;
const W: usize = 50;
const WARMUP: usize = 100;
/// Number of points collected before the initial k-means fit.
const WARMUP_LEN: usize = W + WARMUP - 1;
const LEARNING_RATE: f32 = 0.007;
const KMEANS_ITERS: usize = 20;

/// A window of W time-steps; outer index = time step, inner = dimension.
type MultivariateWindow = Vec<Vec<f32>>;

#[cfg(test)]
mod tests;

/// Streaming K-Means anomaly detector.
///
/// Maintains K centroids over sliding windows of length W. During a warmup
/// phase the detector collects `WARMUP_LEN` points and initialises centroids
/// via k-means++ seeding followed by Lloyd iterations. After warmup, each
/// point is scored as the Euclidean distance from the current window to the
/// nearest centroid; that centroid is then nudged toward the window at
/// `LEARNING_RATE`.
struct KMeans {
    /// Number of input dimensions, fixed at construction.
    dim: usize,
    /// K cluster centroids, each a window of shape (W, dim).
    centroids: Vec<MultivariateWindow>,
    /// Sliding window of the last W points, used for scoring.
    buffer: VecDeque<Vec<f32>>,
    /// Accumulates points during warmup; freed after initialisation.
    warmup_buffer: Vec<Vec<f32>>,
    /// True once initialisation has been attempted.
    initialized: bool,
}

impl KMeans {
    /// Euclidean distance between two windows of identical shape.
    fn window_distance(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(ap, bp)| (ap - bp).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Returns `(centroid_index, distance)` for the centroid nearest to `window`.
    fn find_closest(&self, window: &[Vec<f32>]) -> (usize, f32) {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, Self::window_distance(c, window)))
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("no centroids")
    }

    /// Extracts all contiguous subsequences of length W from `data`.
    fn extract_windows(data: &[Vec<f32>]) -> Vec<MultivariateWindow> {
        let n = data.len();
        if n < W {
            return vec![];
        }
        (0..=n - W).map(|i| data[i..i + W].to_vec()).collect()
    }

    /// Samples one index proportional to `weights` (k-means++ D² step).
    ///
    /// Callers must pass pre-squared distances. Falls back to uniform sampling
    /// when the total weight is zero.
    fn sample_weighted(weights: &[f32], rng: &mut impl Rng) -> usize {
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return rng.random_range(0..weights.len());
        }
        let threshold = rng.random::<f32>() * total;
        let mut cumsum = 0.0;
        for (i, &w) in weights.iter().enumerate() {
            cumsum += w;
            if cumsum >= threshold {
                return i;
            }
        }
        weights.len() - 1
    }

    /// Seeds K centroids via k-means++ then refines them with Lloyd iterations.
    ///
    /// Empty clusters are reseeded from the window most distant from its
    /// current assignment rather than left unchanged.
    fn initialize(&mut self) {
        let windows = Self::extract_windows(&self.warmup_buffer);
        if windows.len() < K {
            return;
        }

        let mut rng = rand::rng();

        // k-means++ seeding: first centroid uniform random, rest D²-weighted.
        let first = rng.random_range(0..windows.len());
        self.centroids = vec![windows[first].clone()];

        while self.centroids.len() < K {
            let weights: Vec<f32> = windows
                .iter()
                .map(|w| {
                    self.centroids
                        .iter()
                        .map(|c| Self::window_distance(c, w))
                        .fold(f32::INFINITY, f32::min)
                        .powi(2)
                })
                .collect();
            let idx = Self::sample_weighted(&weights, &mut rng);
            self.centroids.push(windows[idx].clone());
        }

        // Lloyd iterations.
        for _ in 0..KMEANS_ITERS {
            // Compute all assignments upfront to avoid re-borrowing self inside the update loop.
            let assignments: Vec<(usize, f32)> =
                windows.iter().map(|w| self.find_closest(w)).collect();

            let mut sums: Vec<MultivariateWindow> =
                (0..K).map(|_| vec![vec![0.0f32; self.dim]; W]).collect();
            let mut counts = [0usize; K];

            for (window, &(idx, _)) in windows.iter().zip(assignments.iter()) {
                counts[idx] += 1;
                for (s_row, w_row) in sums[idx].iter_mut().zip(window.iter()) {
                    for (s, &v) in s_row.iter_mut().zip(w_row.iter()) {
                        *s += v;
                    }
                }
            }

            for k in 0..K {
                if counts[k] > 0 {
                    let n = counts[k] as f32;
                    for (c_row, s_row) in self.centroids[k].iter_mut().zip(sums[k].iter()) {
                        for (c, &s) in c_row.iter_mut().zip(s_row.iter()) {
                            *c = s / n;
                        }
                    }
                } else {
                    // Reseed empty centroid from the window furthest from its assignment.
                    if let Some((best_w, _)) = windows
                        .iter()
                        .zip(assignments.iter())
                        .max_by(|(_, (_, da)), (_, (_, db))| da.total_cmp(db))
                    {
                        self.centroids[k] = best_w.clone();
                    }
                }
            }
        }
    }
}

impl Detector for KMeans {
    fn name() -> &'static str {
        "KMeans"
    }

    /// Creates a new detector for `n_dimensions`-dimensional input.
    fn new(n_dimensions: usize) -> Self {
        Self {
            dim: n_dimensions,
            centroids: Vec::with_capacity(K),
            buffer: VecDeque::with_capacity(W),
            warmup_buffer: Vec::with_capacity(WARMUP_LEN),
            initialized: false,
        }
    }

    /// Scores `point` as the Euclidean distance to the nearest centroid window.
    ///
    /// Returns `NaN` for non-finite inputs, during warmup, and until the
    /// sliding window is full. The score is measured against the pre-update
    /// centroid; the nearest centroid is then nudged toward the current window.
    fn update(&mut self, point: &[f32]) -> f32 {
        // Reject non-finite inputs to prevent NaN poisoning of centroids.
        if point.iter().any(|v| !v.is_finite()) {
            return f32::NAN;
        }

        self.buffer.push_back(point.to_vec());
        if self.buffer.len() > W {
            self.buffer.pop_front();
        }

        if !self.initialized {
            self.warmup_buffer.push(point.to_vec());
            if self.warmup_buffer.len() >= WARMUP_LEN {
                self.initialize();
                self.initialized = true;
                self.warmup_buffer = Vec::new();
            }
            return f32::NAN;
        }

        if self.centroids.is_empty() || self.buffer.len() < W {
            return f32::NAN;
        }

        let window: MultivariateWindow = self.buffer.iter().cloned().collect();
        let (idx, dist) = self.find_closest(&window);

        for (cp, wp) in self.centroids[idx].iter_mut().zip(window.iter()) {
            for (c, &w) in cp.iter_mut().zip(wp.iter()) {
                *c += LEARNING_RATE * (w - *c);
            }
        }

        dist
    }
}

touchstone_main!(KMeans);
