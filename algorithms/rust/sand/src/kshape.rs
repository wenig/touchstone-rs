use crate::distance::{dominant_eigenvec, msbd, outer_add, znorm_multi};

pub const KSHAPE_ITER: usize = 5;
pub const POWER_ITER: usize = 20;

/// Output of a k-Shape clustering run.
pub struct KShapeResult {
    /// Cluster centroids; each is a `(n, d)` row-major flat array.
    pub centroids: Vec<Vec<f32>>,
    /// Per-cluster mean mSBD (used as the matching threshold τ).
    pub taus: Vec<f32>,
    /// Per-cluster, per-channel accumulated outer-product matrices S = Σ zₜzₜᵀ.
    /// Layout: `s_matrices[cluster][channel]` is an `n×n` flat row-major slice.
    pub s_matrices: Vec<Vec<Vec<f32>>>,
    /// Number of subsequences assigned to each cluster.
    pub sizes: Vec<usize>,
}

/// k-Shape clustering on multivariate subsequences.
///
/// Each entry in `subseqs` is a `(n, d)` row-major flattened array.
/// Runs up to [`KSHAPE_ITER`] iterations; stops early when assignments and
/// centroids both stabilise. Returns centroids, per-cluster S matrices,
/// mean-distance thresholds τ, and cluster sizes.
pub fn kshape(subseqs: &[Vec<f32>], n: usize, d: usize, k: usize) -> KShapeResult {
    let m = subseqs.len();
    let k = k.min(m);

    // Pre-compute z-normalised subsequences once; reused across all iterations
    // and in the final pass (avoids recomputing znorm_multi O(iter * m) times).
    let znormed: Vec<Vec<f32>> = subseqs.iter().map(|s| znorm_multi(s, n, d)).collect();

    let mut centroids: Vec<Vec<f32>> = (0..k).map(|i| znormed[(i * m) / k].clone()).collect();
    let mut assignments = vec![0usize; m];

    // Reusable per-channel column buffer for outer_add calls.
    let mut col = vec![0.0f32; n];

    for _iter in 0..KSHAPE_ITER {
        let prev = assignments.clone();

        // Assignment: closest centroid by msbd.
        for (idx, s) in subseqs.iter().enumerate() {
            let dists: Vec<f32> = (0..k).map(|ci| msbd(s, &centroids[ci], n, d)).collect();
            assignments[idx] = dists
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        // Centroid update: per-channel dominant eigenvector of S = Σ zₜzₜᵀ.
        let mut any_changed = false;
        for (ci, centroid) in centroids.iter_mut().enumerate() {
            let mut s_mats: Vec<Vec<f32>> = vec![vec![0.0f32; n * n]; d];
            let mut count = 0usize;
            for (zn, &a) in znormed.iter().zip(assignments.iter()) {
                if a != ci {
                    continue;
                }
                for ch in 0..d {
                    for t in 0..n {
                        col[t] = zn[t * d + ch];
                    }
                    outer_add(&mut s_mats[ch], &col);
                }
                count += 1;
            }
            if count == 0 {
                continue;
            }
            let new_c = build_centroid(&s_mats, n, d, centroid);
            let diff: f32 = new_c
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            if diff > 1e-6 {
                any_changed = true;
            }
            *centroid = new_c;
        }

        if !any_changed && assignments == prev {
            break;
        }
    }

    // Final pass: build per-cluster S matrices, taus, sizes.
    // Reuses `znormed` computed above — no duplicate znorm_multi calls.
    let mut s_matrices: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; n * n]; d]; k];
    let mut sizes = vec![0usize; k];
    let mut tau_sums = vec![0.0f32; k];

    for (zn, (s, &ci)) in znormed.iter().zip(subseqs.iter().zip(assignments.iter())) {
        for ch in 0..d {
            for t in 0..n {
                col[t] = zn[t * d + ch];
            }
            outer_add(&mut s_matrices[ci][ch], &col);
        }
        sizes[ci] += 1;
        tau_sums[ci] += msbd(s, &centroids[ci], n, d);
    }

    let taus = (0..k)
        .map(|i| {
            if sizes[i] > 0 {
                tau_sums[i] / sizes[i] as f32
            } else {
                0.0
            }
        })
        .collect();

    KShapeResult {
        centroids,
        taus,
        s_matrices,
        sizes,
    }
}

/// Assemble a `(n, d)` centroid from per-channel dominant eigenvectors of `s_mats`.
///
/// Computes the dominant eigenvector of each `n×n` channel matrix via power
/// iteration, writes it into the corresponding column of the output, then flips
/// the sign of the whole centroid if it points away from `prev` (dot product < 0).
pub fn build_centroid(s_mats: &[Vec<f32>], n: usize, d: usize, prev: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0f32; n * d];
    for ch in 0..d {
        let ev = dominant_eigenvec(&s_mats[ch], n);
        for t in 0..n {
            c[t * d + ch] = ev[t];
        }
    }
    let dot: f32 = c.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
    if dot < 0.0 {
        c.iter_mut().for_each(|v| *v = -*v);
    }
    c
}
