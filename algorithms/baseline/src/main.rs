//! Random-score baseline detector.

use touchstone_rs::{Detector, touchstone_main};

pub struct BaselineDetector;

impl Detector for BaselineDetector {
    fn name() -> &'static str {
        "Baseline"
    }

    fn new(_n_dimensions: usize) -> Self {
        Self
    }

    fn update(&mut self, _point: &[f32]) -> f32 {
        rand::random()
    }
}

touchstone_main!(BaselineDetector);
