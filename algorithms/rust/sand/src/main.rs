mod distance;
mod kshape;

#[cfg(test)]
mod tests;

use std::collections::VecDeque;

use distance::msbd;
use kshape::{KShapeResult, build_centroid, kshape};
use touchstone_rs::{Detector, touchstone_main};

const L: usize = 100;
const L_THETA_MULT: usize = 1;
const K: usize = 3;
const BSIZE: usize = 300;
const ALPHA: f32 = 0.5;

// ── model ────────────────────────────────────────────────────────────────────

/// A single cluster maintained by the online SAND model.
struct ClusterEntry {
    /// Cluster centroid; `(l_theta, d)` row-major flat array.
    centroid: Vec<f32>,
    /// Per-channel accumulated S = Σ zₜzₜᵀ matrices; `s_matrices[ch]` is `l_theta²`.
    s_matrices: Vec<Vec<f32>>,
    /// Mean mSBD of cluster members — used as the matching threshold.
    tau: f32,
    /// Normalised mixture weight (sums to 1 across the model).
    weight: f32,
    /// Number of subsequences ever assigned to this cluster.
    size: usize,
    /// `point_count` at the last batch that touched this cluster.
    t_last: usize,
}

// ── detector ─────────────────────────────────────────────────────────────────

/// Streaming Anomaly with Normalised Distances (SAND) detector.
///
/// Maintains an online mixture model of `k` k-Shape centroids and scores
/// each incoming time step by the weighted minimum mSBD to the model.
/// The model is updated in batches of `bsize` points.
pub struct SAND {
    l: usize,
    l_theta: usize,
    k: usize,
    bsize: usize,
    alpha: f32,
    n_dims: usize,

    stream: VecDeque<Vec<f32>>, // sliding window — last l_theta time steps
    batch: Vec<Vec<f32>>,       // accumulates current in-flight batch
    model: Vec<ClusterEntry>,

    point_count: usize,
    initialized: bool,

    score_mean: f32,
    score_std: f32,
}

impl SAND {
    fn extract_subseqs(data: &[Vec<f32>], l: usize) -> Vec<Vec<f32>> {
        if data.len() < l {
            return vec![];
        }
        (0..=data.len() - l)
            .map(|i| (0..l).flat_map(|t| data[i + t].iter().cloned()).collect())
            .collect()
    }

    fn initialize(&mut self, data: &[Vec<f32>]) {
        let subseqs = Self::extract_subseqs(data, self.l_theta);
        if subseqs.len() < self.k {
            return;
        }
        let res = kshape(&subseqs, self.l_theta, self.n_dims, self.k);
        let k = res.centroids.len();
        let raw_w = self.raw_weights(&res.centroids, &res.sizes);
        let w_sum: f32 = raw_w.iter().sum::<f32>().max(1e-8);

        self.model = (0..k)
            .map(|i| ClusterEntry {
                centroid: res.centroids[i].clone(),
                s_matrices: res.s_matrices[i].clone(),
                tau: res.taus[i],
                weight: raw_w[i] / w_sum,
                size: res.sizes[i],
                t_last: self.point_count,
            })
            .collect();

        self.initialized = true;
    }

    fn batch_update(&mut self, data: &[Vec<f32>]) {
        let subseqs = Self::extract_subseqs(data, self.l_theta);
        if subseqs.len() < self.k {
            return;
        }
        let KShapeResult {
            centroids: new_centroids,
            taus: new_taus,
            s_matrices: new_s,
            sizes: new_sizes,
        } = kshape(&subseqs, self.l_theta, self.n_dims, self.k);
        let l_theta = self.l_theta;
        let d = self.n_dims;

        for ci in 0..new_centroids.len() {
            let match_idx = self
                .model
                .iter()
                .enumerate()
                .find(|(_, e)| msbd(&e.centroid, &new_centroids[ci], l_theta, d) < e.tau)
                .map(|(i, _)| i);

            if let Some(idx) = match_idx {
                for (dst, src) in self.model[idx].s_matrices.iter_mut().zip(new_s[ci].iter()) {
                    for (a, b) in dst.iter_mut().zip(src.iter()) {
                        *a += b;
                    }
                }
                let prev_c = self.model[idx].centroid.clone();
                self.model[idx].centroid =
                    build_centroid(&self.model[idx].s_matrices, l_theta, d, &prev_c);
                let n_old = self.model[idx].size as f32;
                let n_new = new_sizes[ci] as f32;
                self.model[idx].tau =
                    (n_old * self.model[idx].tau + n_new * new_taus[ci]) / (n_old + n_new);
                self.model[idx].size += new_sizes[ci];
                self.model[idx].t_last = self.point_count;
            } else {
                self.model.push(ClusterEntry {
                    centroid: new_centroids[ci].clone(),
                    s_matrices: new_s[ci].clone(),
                    tau: new_taus[ci],
                    weight: 0.0,
                    size: new_sizes[ci],
                    t_last: self.point_count,
                });
            }
        }

        let centroids: Vec<Vec<f32>> = self.model.iter().map(|e| e.centroid.clone()).collect();
        let sizes: Vec<usize> = self.model.iter().map(|e| e.size).collect();
        let raw_w = self.raw_weights(&centroids, &sizes);

        for (i, entry) in self.model.iter_mut().enumerate() {
            let a_i = self.point_count.saturating_sub(entry.t_last);
            let decay = 1.0 / (a_i.saturating_sub(self.bsize).max(1)) as f32;
            entry.weight = (1.0 - self.alpha) * entry.weight + self.alpha * raw_w[i] * decay;
        }

        let w_sum: f32 = self.model.iter().map(|e| e.weight).sum::<f32>().max(1e-8);
        for e in self.model.iter_mut() {
            e.weight /= w_sum;
        }
    }

    fn raw_weights(&self, centroids: &[Vec<f32>], sizes: &[usize]) -> Vec<f32> {
        let n = centroids.len();
        let l_theta = self.l_theta;
        let d = self.n_dims;
        (0..n)
            .map(|i| {
                let dsum: f32 = (0..n)
                    .map(|j| msbd(&centroids[i], &centroids[j], l_theta, d))
                    .sum::<f32>()
                    .max(1e-8);
                (sizes[i] as f32).powi(2) / dsum
            })
            .collect()
    }

    fn score_subseq(&self, subseq: &[f32]) -> f32 {
        let l = self.l;
        let d = self.n_dims;
        self.model
            .iter()
            .map(|e| {
                let l_theta = self.l_theta;
                if l_theta < l {
                    return 0.0;
                }
                // Slide a (l, d) window across the (l_theta, d) centroid.
                // The centroid is contiguous row-major, so each window is a
                // plain slice — no copy needed.
                let min_d = (0..=l_theta - l)
                    .map(|x| msbd(subseq, &e.centroid[x * d..(x + l) * d], l, d))
                    .fold(f32::INFINITY, f32::min);
                e.weight * min_d
            })
            .sum()
    }
}

impl Detector for SAND {
    fn name() -> &'static str {
        "SAND"
    }

    fn new(n_dimensions: usize) -> Self {
        let n_dims = n_dimensions.max(1);
        Self {
            l: L,
            l_theta: L * L_THETA_MULT,
            k: K,
            bsize: BSIZE,
            alpha: ALPHA,
            n_dims,
            stream: VecDeque::new(),
            batch: Vec::with_capacity(BSIZE),
            model: Vec::new(),
            point_count: 0,
            initialized: false,
            score_mean: 0.0,
            score_std: 1.0,
        }
    }

    fn update(&mut self, point: &[f32]) -> f32 {
        self.point_count += 1;
        let p = point.to_vec();
        self.stream.push_back(p.clone());
        self.batch.push(p);

        while self.stream.len() > self.l_theta {
            self.stream.pop_front();
        }

        if self.batch.len() >= self.bsize {
            let data = std::mem::take(&mut self.batch);
            if !self.initialized {
                self.initialize(&data);
            } else {
                self.batch_update(&data);
            }
        }

        if !self.initialized || self.stream.len() < self.l {
            return f32::NAN;
        }

        // Score the subsequence formed by the last `l` time steps.
        let n = self.stream.len();
        let l = self.l;
        let subseq: Vec<f32> = (n - l..n)
            .flat_map(|i| self.stream[i].iter().cloned())
            .collect();

        let raw = self.score_subseq(&subseq);

        // Exponential moving average normalisation.
        self.score_mean = self.alpha * raw + (1.0 - self.alpha) * self.score_mean;
        let diff = (raw - self.score_mean).abs();
        self.score_std = self.alpha * diff + (1.0 - self.alpha) * self.score_std;

        if self.score_std < 1e-8 {
            raw
        } else {
            (raw - self.score_mean) / self.score_std
        }
    }
}

touchstone_main!(SAND);
