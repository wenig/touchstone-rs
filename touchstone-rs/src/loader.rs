use anyhow::{Context, Result};
use polars::prelude::*;
use std::fs::File;
use std::path::Path;

/// In-memory representation of one benchmark dataset.
pub struct Dataset {
    /// Dataset identifier, usually derived from the file stem.
    pub name: String,
    /// Feature matrix in row-major layout (`n_points x n_dims`).
    pub features: Vec<Vec<f32>>,
    /// Binary anomaly labels aligned with `features`.
    pub labels: Vec<u8>,
}

/// Returns sorted `(name, path)` pairs without loading any data.
///
/// Parquet files take priority over CSV files when both share the same stem.
pub fn list_datasets(dir: &Path) -> Result<Vec<(String, std::path::PathBuf)>> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read data dir: {}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|x| x == "parquet" || x == "csv")
                .unwrap_or(false)
        })
        .collect();

    entries.sort_by_key(|e| e.path());

    // Deduplicate by stem, preferring parquet over csv.
    let mut seen: std::collections::HashMap<String, std::path::PathBuf> =
        std::collections::HashMap::new();
    for e in entries {
        let path = e.path();
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let is_parquet = path.extension().map(|x| x == "parquet").unwrap_or(false);
        if is_parquet || !seen.contains_key(&stem) {
            seen.insert(stem, path);
        }
    }

    let mut result: Vec<_> = seen.into_iter().collect();
    result.sort_by(|a, b| a.1.cmp(&b.1));

    Ok(result)
}

/// Loads a dataset from a Parquet or CSV file, applying the given name to the result.
pub fn load_dataset(name: String, path: &Path) -> Result<Dataset> {
    let loader = if path.extension().map(|x| x == "parquet").unwrap_or(false) {
        load_parquet
    } else {
        load_csv
    };
    loader(path)
        .with_context(|| format!("failed to load {}", path.display()))
        .map(|ds| Dataset { name, ..ds })
}

/// Parses a Parquet file into a Dataset.
///
/// Expects format: `timestamp, feature_1, ..., feature_n, label`
fn load_parquet(path: &Path) -> Result<Dataset> {
    let file = File::open(path).context("open parquet file")?;
    let df = ParquetReader::new(file).finish().context("parquet parse")?;

    extract_dataset(df)
}

/// Parses a CSV file into a Dataset.
///
/// Expects format: `timestamp, feature_1, ..., feature_n, label`
fn load_csv(path: &Path) -> Result<Dataset> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.into()))
        .context("csv reader init")?
        .finish()
        .context("csv parse")?;

    extract_dataset(df)
}

/// Shared extraction logic for both CSV and Parquet DataFrames.
///
/// Expects columns: timestamp (skipped), feature_1..feature_n, label (last).
fn extract_dataset(df: DataFrame) -> Result<Dataset> {
    let n_rows = df.height();
    let n_cols = df.width();
    anyhow::ensure!(
        n_cols >= 3,
        "dataset must have timestamp, at least one feature column, and one label column"
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
    use polars::prelude::{Column, DataFrame, ParquetWriter};
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    fn make_df() -> DataFrame {
        DataFrame::new(
            3,
            vec![
                Column::new("timestamp".into(), &[0i64, 1, 2]),
                Column::new("x".into(), &[1.0f64, 3.0, 5.0]),
                Column::new("y".into(), &[2.0f64, 4.0, 6.0]),
                Column::new("label".into(), &[0i64, 1, 0]),
            ],
        )
        .unwrap()
    }

    fn write_parquet(df: &mut DataFrame) -> NamedTempFile {
        let f = NamedTempFile::with_suffix(".parquet").unwrap();
        let out = std::fs::File::create(f.path()).unwrap();
        ParquetWriter::new(out).finish(df).unwrap();
        f
    }

    #[test]
    fn parse_simple_csv() {
        let mut f = NamedTempFile::with_suffix(".csv").unwrap();
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

    #[test]
    fn parse_simple_parquet() {
        let mut df = make_df();
        let f = write_parquet(&mut df);

        let ds = load_parquet(f.path()).unwrap();
        assert_eq!(ds.labels, vec![0, 1, 0]);
        assert_eq!(ds.features.len(), 3);
        assert_eq!(ds.features[0], vec![1.0, 2.0]);
        assert_eq!(ds.features[1], vec![3.0, 4.0]);
    }

    #[test]
    fn load_dataset_dispatches_by_extension() {
        let mut df = make_df();
        let f = write_parquet(&mut df);

        let ds = load_dataset("test".into(), f.path()).unwrap();
        assert_eq!(ds.name, "test");
        assert_eq!(ds.labels, vec![0, 1, 0]);
    }

    #[test]
    fn list_datasets_parquet_preferred_over_csv() {
        let dir = TempDir::new().unwrap();
        let stem = "mydata";

        // Write CSV
        let csv_path = dir.path().join(format!("{stem}.csv"));
        std::fs::write(&csv_path, "timestamp,x,label\n0,9.0,1\n1,8.0,1\n2,7.0,1\n").unwrap();

        // Write Parquet with different content so we can tell which was loaded
        let mut df = make_df();
        let parquet_path = dir.path().join(format!("{stem}.parquet"));
        let out = std::fs::File::create(&parquet_path).unwrap();
        ParquetWriter::new(out).finish(&mut df).unwrap();

        let datasets = list_datasets(dir.path()).unwrap();
        assert_eq!(datasets.len(), 1, "duplicate stems should be deduplicated");
        assert_eq!(datasets[0].0, stem);
        assert_eq!(datasets[0].1.extension().unwrap(), "parquet");
    }

    #[test]
    fn list_datasets_falls_back_to_csv() {
        let dir = TempDir::new().unwrap();
        let csv_path = dir.path().join("only.csv");
        std::fs::write(&csv_path, "timestamp,x,label\n0,1.0,0\n").unwrap();

        let datasets = list_datasets(dir.path()).unwrap();
        assert_eq!(datasets.len(), 1);
        assert_eq!(datasets[0].1.extension().unwrap(), "csv");
    }

    #[test]
    fn list_datasets_sorted_by_path() {
        let dir = TempDir::new().unwrap();
        for name in ["c_data", "a_data", "b_data"] {
            std::fs::write(dir.path().join(format!("{name}.csv")), "t,x,l\n0,1.0,0\n").unwrap();
        }

        let datasets = list_datasets(dir.path()).unwrap();
        let names: Vec<&str> = datasets.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["a_data", "b_data", "c_data"]);
    }

    #[test]
    fn rejects_too_few_columns() {
        let mut f = NamedTempFile::with_suffix(".csv").unwrap();
        writeln!(f, "timestamp,label").unwrap();
        writeln!(f, "0,0").unwrap();

        assert!(load_csv(f.path()).is_err());
    }
}
