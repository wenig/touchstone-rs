use std::path::PathBuf;

use polars::io::SerWriter;
use polars::prelude::IpcWriter;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use touchstone_rs::{Detector, DetectorFactory, Touchstone};

struct PythonDetectorInstance {
    obj: Py<PyAny>,
}

impl Detector for PythonDetectorInstance {
    fn name() -> &'static str {
        "python_detector"
    }

    fn new(_n_dimensions: usize) -> Self {
        unimplemented!("instantiated via PythonDetectorFactory::create")
    }

    fn update(&mut self, point: &[f32]) -> f32 {
        Python::try_attach(|py| {
            self.obj
                .call_method1(py, "update", (point.to_vec(),))
                .and_then(|r| r.extract::<f32>(py))
                .unwrap_or(f32::NAN)
        })
        .unwrap_or(f32::NAN)
    }
}

/// Holds a Python detector class and creates fresh instances per dataset.
struct PythonDetectorFactory {
    cls: Py<PyAny>,
    detector_name: String,
}

impl DetectorFactory for PythonDetectorFactory {
    fn name(&self) -> String {
        self.detector_name.clone()
    }

    fn create(&self, n_dims: usize) -> Box<dyn Detector> {
        let obj = Python::try_attach(|py| {
            self.cls
                .call1(py, (n_dims,))
                .expect("failed to instantiate Python detector")
        })
        .expect("Python interpreter not attached");
        Box::new(PythonDetectorInstance { obj })
    }
}

/// Runs Touchstone evaluation against the given Python detector classes.
///
/// Each element of `algorithms` must be a callable (class) that:
/// - accepts `n_dimensions: int` as its sole constructor argument
/// - exposes a `name() -> str` classmethod or staticmethod
/// - exposes an `update(point: list[float]) -> float` instance method
///
/// Returns the results as Arrow IPC bytes. Use `polars.read_ipc(io.BytesIO(result))`
/// or the `touchstone_py.run_touchstone` wrapper to get a `polars.DataFrame`.
#[pyfunction]
fn run_touchstone<'py>(
    py: Python<'py>,
    data_dir: PathBuf,
    algorithms: Vec<Py<PyAny>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut ts = Touchstone::new(&data_dir);

    for cls in algorithms {
        let detector_name: String = cls.call_method0(py, "name")?.extract(py)?;
        ts.add_detector_factory(Box::new(PythonDetectorFactory { cls, detector_name }));
    }

    let mut df = ts
        .run()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let mut buf: Vec<u8> = Vec::new();
    IpcWriter::new(&mut buf)
        .finish(&mut df)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(PyBytes::new(py, &buf))
}

#[pymodule]
fn _rust_backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_touchstone, m)?)?;
    Ok(())
}
