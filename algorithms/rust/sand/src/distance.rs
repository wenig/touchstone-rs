use std::cell::RefCell;

use rustfft::{FftPlanner, num_complex::Complex};

use crate::kshape::POWER_ITER;

// ── thread-local scratch ─────────────────────────────────────────────────────
//
// Two separate `RefCell`s so we can borrow them independently inside `msbd`:
// `CHAN` holds the per-channel z-normalised column buffers (read after fill),
// `FFT` holds the accumulated NCC sum and the complex FFT buffers (written).
// Keeping them apart avoids the borrow-checker conflict that would arise from
// borrowing multiple fields of a single struct mutably at the same time.

struct ChanScratch {
    xch: Vec<f32>,
    ych: Vec<f32>,
}

struct FftScratch {
    /// Accumulated NCC sum across channels, length `2n - 1`.
    summed: Vec<f32>,
    /// Complex forward/inverse FFT work buffer, length = next power-of-two ≥ `2n-1`.
    buf_x: Vec<Complex<f32>>,
    buf_y: Vec<Complex<f32>>,
}

impl FftScratch {
    /// Returns mutable slices for all three buffers in one call so the borrow
    /// checker can see they are disjoint fields.
    fn parts_mut(
        &mut self,
        cc_len: usize,
        fft_size: usize,
    ) -> (&mut [f32], &mut [Complex<f32>], &mut [Complex<f32>]) {
        (
            &mut self.summed[..cc_len],
            &mut self.buf_x[..fft_size],
            &mut self.buf_y[..fft_size],
        )
    }
}

thread_local! {
    static PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
    static CHAN: RefCell<ChanScratch> = const {
        RefCell::new(ChanScratch {
            xch: Vec::new(),
            ych: Vec::new(),
        })
    };
    static FFT: RefCell<FftScratch> = const {
        RefCell::new(FftScratch {
            summed: Vec::new(),
            buf_x: Vec::new(),
            buf_y: Vec::new(),
        })
    };
}

// ── z-normalisation ───────────────────────────────────────────────────────────

/// Z-normalise a 1-D slice using population standard deviation.
///
/// Returns an all-zeros vector when `std < 1e-8` (constant input).
#[allow(dead_code)]
pub fn znorm(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    if std < 1e-8 {
        vec![0.0; x.len()]
    } else {
        x.iter().map(|&v| (v - mean) / std).collect()
    }
}

/// Z-normalise each channel of a `(n, d)` row-major array independently.
///
/// Each channel is standardised to zero mean and unit population variance.
/// Constant channels (std < 1e-8) are set to zero.
pub fn znorm_multi(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for ch in 0..d {
        let mean = (0..n).map(|t| x[t * d + ch]).sum::<f32>() / n as f32;
        let var = (0..n).map(|t| (x[t * d + ch] - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt();
        for t in 0..n {
            out[t * d + ch] = if std < 1e-8 {
                0.0
            } else {
                (x[t * d + ch] - mean) / std
            };
        }
    }
    out
}

// ── FFT cross-correlation ─────────────────────────────────────────────────────

/// Accumulate the FFT-based normalised cross-correlation for one z-normalised
/// channel pair into `summed`.
///
/// All buffers are caller-supplied so no heap allocation occurs inside this
/// function. `buf_x` and `buf_y` must have length ≥ `next_power_of_two(2n-1)`.
/// `summed` must have length `2n - 1`. The result covers lags `[-(n-1), n-1]`
/// with negative lags first (same layout as [`ncc_c`]).
fn ncc_c_add_into(
    xch: &[f32],
    ych: &[f32],
    summed: &mut [f32],
    buf_x: &mut [Complex<f32>],
    buf_y: &mut [Complex<f32>],
) {
    let n = xch.len();
    let fft_size = (2 * n - 1).next_power_of_two();

    for (i, &v) in xch.iter().enumerate() {
        buf_x[i] = Complex::new(v, 0.0);
    }
    for c in buf_x[n..fft_size].iter_mut() {
        *c = Complex::new(0.0, 0.0);
    }
    for (i, &v) in ych.iter().enumerate() {
        buf_y[i] = Complex::new(v, 0.0);
    }
    for c in buf_y[n..fft_size].iter_mut() {
        *c = Complex::new(0.0, 0.0);
    }

    PLANNER.with(|p| {
        let mut planner = p.borrow_mut();
        let fwd = planner.plan_fft_forward(fft_size);
        fwd.process(&mut buf_x[..fft_size]);
        fwd.process(&mut buf_y[..fft_size]);
        for (bx, by) in buf_x[..fft_size].iter_mut().zip(buf_y[..fft_size].iter()) {
            *bx *= by.conj();
        }
        let inv = planner.plan_fft_inverse(fft_size);
        inv.process(&mut buf_x[..fft_size]);
    });

    let scale = 1.0 / fft_size as f32;
    let denom =
        (xch.iter().map(|v| v * v).sum::<f32>() * ych.iter().map(|v| v * v).sum::<f32>()).sqrt();
    if denom < 1e-10 {
        return;
    }
    let norm = scale / denom;

    // Rearrange circular output: negative lags (tail of IFFT) first, then non-negative.
    if n > 1 {
        for (s, c) in summed[..n - 1]
            .iter_mut()
            .zip(buf_x[fft_size - (n - 1)..fft_size].iter())
        {
            *s += c.re * norm;
        }
    }
    for (s, c) in summed[n - 1..].iter_mut().zip(buf_x[..n].iter()) {
        *s += c.re * norm;
    }
}

/// FFT-based normalised cross-correlation for a single channel.
///
/// Returns a vector of length `2n - 1` covering all lags `[-(n-1), n-1]`
/// with negative lags first (index `n-1` corresponds to lag 0).
/// Values are in `[-1, 1]`.
///
/// Hot paths call [`msbd`] directly which avoids returning a `Vec`; this
/// function is provided for testing and one-off use.
#[allow(dead_code)]
pub fn ncc_c(x: &[f32], y: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let cc_len = 2 * n - 1;
    let fft_size = cc_len.next_power_of_two();

    FFT.with(|fft| {
        let mut fft = fft.borrow_mut();
        fft.summed.resize(cc_len, 0.0);
        for v in fft.summed[..cc_len].iter_mut() {
            *v = 0.0;
        }
        if fft.buf_x.len() < fft_size {
            fft.buf_x.resize(fft_size, Complex::new(0.0, 0.0));
            fft.buf_y.resize(fft_size, Complex::new(0.0, 0.0));
        }
        let (summed, buf_x, buf_y) = fft.parts_mut(cc_len, fft_size);
        ncc_c_add_into(x, y, summed, buf_x, buf_y);
        fft.summed[..cc_len].to_vec()
    })
}

// ── distance ─────────────────────────────────────────────────────────────────

/// Multivariate Shape-Based Distance (mSBD).
///
/// Sums the normalised cross-correlation curves across all `d` channels to
/// find a consensus lag, then returns `1 - max_NCC / d`. The result is in
/// `[0, 2]`; 0 means identical shape, values near 2 mean anti-correlated.
///
/// `x` and `y` must be `(n, d)` row-major flat arrays. Channels are
/// z-normalised internally. Uses thread-local scratch buffers to avoid
/// per-call heap allocation.
pub fn msbd(x: &[f32], y: &[f32], n: usize, d: usize) -> f32 {
    let cc_len = 2 * n - 1;
    let fft_size = cc_len.next_power_of_two();

    FFT.with(|fft| {
        let mut fft = fft.borrow_mut();
        fft.summed.resize(cc_len, 0.0);
        for v in fft.summed[..cc_len].iter_mut() {
            *v = 0.0;
        }
        if fft.buf_x.len() < fft_size {
            fft.buf_x.resize(fft_size, Complex::new(0.0, 0.0));
            fft.buf_y.resize(fft_size, Complex::new(0.0, 0.0));
        }

        CHAN.with(|chan| {
            let mut chan = chan.borrow_mut();
            if chan.xch.len() < n {
                chan.xch.resize(n, 0.0);
                chan.ych.resize(n, 0.0);
            }

            for ch in 0..d {
                let mean_x = (0..n).map(|t| x[t * d + ch]).sum::<f32>() / n as f32;
                let var_x = (0..n)
                    .map(|t| (x[t * d + ch] - mean_x).powi(2))
                    .sum::<f32>()
                    / n as f32;
                let std_x = var_x.sqrt();

                let mean_y = (0..n).map(|t| y[t * d + ch]).sum::<f32>() / n as f32;
                let var_y = (0..n)
                    .map(|t| (y[t * d + ch] - mean_y).powi(2))
                    .sum::<f32>()
                    / n as f32;
                let std_y = var_y.sqrt();

                for t in 0..n {
                    chan.xch[t] = if std_x < 1e-8 {
                        0.0
                    } else {
                        (x[t * d + ch] - mean_x) / std_x
                    };
                    chan.ych[t] = if std_y < 1e-8 {
                        0.0
                    } else {
                        (y[t * d + ch] - mean_y) / std_y
                    };
                }

                let (summed, buf_x, buf_y) = fft.parts_mut(cc_len, fft_size);
                ncc_c_add_into(&chan.xch[..n], &chan.ych[..n], summed, buf_x, buf_y);
            }
        });

        let best = fft.summed[..cc_len]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(n - 1);
        let max_ncc = fft.summed[best] / d as f32;
        (1.0 - max_ncc).max(0.0)
    })
}

// ── linear algebra ────────────────────────────────────────────────────────────

/// Accumulate the outer product `v·vᵀ` into `s`.
///
/// `s` must be an `n×n` row-major flat slice (`s.len() == n²`).
/// The hoisted `vi` scalar allows the compiler to auto-vectorise the inner loop.
pub fn outer_add(s: &mut [f32], v: &[f32]) {
    let n = v.len();
    for i in 0..n {
        let vi = v[i];
        for j in 0..n {
            s[i * n + j] += vi * v[j];
        }
    }
}

/// Dominant eigenvector of a symmetric `n×n` matrix `m` via power iteration.
///
/// Runs [`POWER_ITER`] steps. Initialises with a uniform vector. The work
/// buffer `mv` is allocated once outside the loop to avoid repeated allocation.
pub fn dominant_eigenvec(m: &[f32], n: usize) -> Vec<f32> {
    let mut v = vec![1.0f32 / (n as f32).sqrt(); n];
    let mut mv = vec![0.0f32; n];
    for _ in 0..POWER_ITER {
        for x in mv.iter_mut() {
            *x = 0.0;
        }
        for i in 0..n {
            for j in 0..n {
                mv[i] += m[i * n + j] * v[j];
            }
        }
        let norm: f32 = mv.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            break;
        }
        for (vi, mvi) in v.iter_mut().zip(mv.iter()) {
            *vi = mvi / norm;
        }
    }
    v
}
