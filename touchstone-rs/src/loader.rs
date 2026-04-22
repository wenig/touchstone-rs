use anyhow::{Context, Result};
use polars::prelude::*;
use std::path::Path;

/// In-memory representation of one benchmark dataset.
pub struct Dataset {
    /// Dataset identifier, usually derived from the CSV file stem.
    pub name: String,
    /// Feature matrix in row-major layout (`n_points x n_dims`).
    pub features: Vec<Vec<f32>>,
    /// Binary anomaly labels aligned with `features`.
    pub labels: Vec<u8>,
}

/// Returns sorted `(name, path)` pairs without loading any data.
pub fn list_datasets(dir: &Path) -> Result<Vec<(String, std::path::PathBuf)>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read data dir: {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "csv").unwrap_or(false))
        .collect();

    entries.sort_by_key(|e| e.path());

    Ok(entries
        .into_iter()
        .map(|e| {
            let path = e.path();
            let name = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            (name, path)
        })
        .collect())
}

/// Loads a dataset from a CSV file, applying the given name to the result.
pub fn load_dataset(name: String, path: &Path) -> Result<Dataset> {
    load_csv(path)
        .with_context(|| format!("failed to load {}", path.display()))
        .map(|ds| Dataset { name, ..ds })
}

/// Parses a CSV file into a Dataset.
///
/// Expects format: `timestamp, feature_1, ..., feature_n, label`
/// The timestamp column is skipped; features are cast to f32; the last column is the binary label.
fn load_csv(path: &Path) -> Result<Dataset> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.into()))
        .context("csv reader init")?
        .finish()
        .context("csv parse")?;

    let n_rows = df.height();
    let n_cols = df.width();
    anyhow::ensure!(
        n_cols >= 3,
        "CSV must have timestamp, at least one feature column, and one label column"
    );

    let cols: &[Column] = df.columns();
    let label_col: &Column = cols.last().unwrap();
    let labels: Vec<u8> = label_col
        .cast(&DataType::Int64)
        .context("label cast")?
        .i64()
        .context("label as i64")?
        .into_iter()
        .map(|v: Option<i64>| v.unwrap_or(0) as u8)
        .collect();

    // cols[0] is timestamp — skip it
    let feature_cols: &[Column] = &cols[1..n_cols - 1];
    let cast_cols: Vec<Column> = feature_cols
        .iter()
        .map(|c: &Column| c.cast(&DataType::Float32).context("feature cast"))
        .collect::<Result<_>>()?;

    let features: Vec<Vec<f32>> = (0..n_rows)
        .map(|i| {
            cast_cols
                .iter()
                .map(|c: &Column| {
                    c.f32()
                        .expect("cast to f32 failed")
                        .get(i)
                        .unwrap_or(f32::NAN)
                })
                .collect()
        })
        .collect();

    Ok(Dataset {
        name: String::new(),
        features,
        labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parse_simple_csv() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "timestamp,x,y,label").unwrap();
        writeln!(f, "0,1.0,2.0,0").unwrap();
        writeln!(f, "1,3.0,4.0,1").unwrap();
        writeln!(f, "2,5.0,6.0,0").unwrap();

        let ds = load_csv(f.path()).unwrap();
        assert_eq!(ds.labels, vec![0, 1, 0]);
        assert_eq!(ds.features.len(), 3);
        assert_eq!(ds.features[0], vec![1.0, 2.0]);
        assert_eq!(ds.features[1], vec![3.0, 4.0]);
    }
}
