//! Example detector using a rolling normal-distribution assumption.
//!
//! For each new point, the detector computes a z-score-like anomaly score from
//! rolling mean/std statistics over a fixed-size warmup window.

use std::{collections::VecDeque, path::Path};
use touchstone_rs::{Detector, Touchstone};

/// Simple multivariate rolling z-score detector.
struct NormalDistributionDetector {
    /// Expected number of dimensions per point.
    n_dimensions: usize,
    /// Sliding window of recent points used to estimate mean/std.
    buffer: VecDeque<Vec<f32>>,
}

impl Detector for NormalDistributionDetector {
    /// Returns `NaN` during warmup, then a summed absolute z-score.
    fn update(&mut self, point: &[f32]) -> f32 {
        if point.len() != self.n_dimensions || self.n_dimensions == 0 {
            return f32::NAN;
        }

        if self.buffer.len() < self.buffer.capacity() {
            self.buffer.push_back(point.to_vec());
            return f32::NAN;
        }

        let n = self.buffer.len() as f32;
        let means: Vec<f32> = (0..self.n_dimensions)
            .map(|dim| self.buffer.iter().map(|p| p[dim]).sum::<f32>() / n)
            .collect();

        let stds: Vec<f32> = (0..self.n_dimensions)
            .map(|dim| {
                let var = self
                    .buffer
                    .iter()
                    .map(|p| {
                        let d = p[dim] - means[dim];
                        d * d
                    })
                    .sum::<f32>()
                    / n;
                var.sqrt().max(1e-6)
            })
            .collect();

        let score = (0..self.n_dimensions)
            .map(|dim| ((point[dim] - means[dim]) / stds[dim]).abs())
            .sum::<f32>();

        self.buffer.pop_front();
        self.buffer.push_back(point.to_vec());

        score
    }
}

fn main() {
    // Run the example against local Touchstone datasets.
    let mut experiment = Touchstone::new(Path::new("data"));
    experiment.add_detector("NormalDistribution-20", |n_dimensions| {
        NormalDistributionDetector {
            n_dimensions,
            buffer: VecDeque::with_capacity(20),
        }
    });
    let report_df = experiment.run().unwrap();
    println!("{report_df}");
}
