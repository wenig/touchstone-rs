//! Random-score baseline detector.

use rand::RngExt;
use rand::rngs::SmallRng;
use touchstone_rs::{Detector, touchstone_main};

pub struct BaselineDetector {
    rng: SmallRng,
}

impl Detector for BaselineDetector {
    fn name() -> &'static str {
        "Baseline"
    }

    fn new(_n_dimensions: usize) -> Self {
        Self { rng: rand::make_rng() }
    }

    fn update(&mut self, _point: &[f32]) -> f32 {
        self.rng.random()
    }
}

touchstone_main!(BaselineDetector);
