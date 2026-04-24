# Benchmark Datasets

This directory contains 20 time series anomaly detection datasets selected from established benchmarks. All datasets are in CSV format with time-value pairs.

## Sources & Citations

### GutenTAG (7 datasets)
Synthetically generated datasets using a novel time series anomaly generator.

- `GutenTAG-rw-channels-all-of-3.parquet`
- `GutenTAG-cbf-position-end.parquet`
- `GutenTAG-ecg-type-pattern-shift.parquet`
- `GutenTAG-poly-channels-single-of-2.parquet`
- `GutenTAG-poly-same-count-1.parquet`
- `GutenTAG-sinus-channels-single-of-10.parquet`
- `GutenTAG-sinus-channels-single-of-5.parquet`

**Citation:** Wenig, P., Schmidl, B., & Papenbrock, T. (2021). TimeEval: A Benchmark for Time Series Anomaly Detection Algorithms. arXiv:2011.13504.

### KDD-TSAD (6 datasets)
Datasets from the KDD Cup 2021 Time Series Anomaly Detection competition.

- `KDD-TSAD-001_UCR_Anomaly_DISTORTED1sddb40.parquet`
- `KDD-TSAD-058_UCR_Anomaly_DISTORTEDapneaecg.parquet`
- `KDD-TSAD-081_UCR_Anomaly_DISTORTEDresperation3.parquet`
- `KDD-TSAD-114_UCR_Anomaly_CIMIS44AirTemperature2.parquet`
- `KDD-TSAD-143_UCR_Anomaly_InternalBleeding8.parquet`
- `KDD-TSAD-189_UCR_Anomaly_resperation3.parquet`

**Citation:** arXiv:2009.13807

### MITDB (1 dataset)
ECG data from the MIT-BIH Arrhythmia Database.

- `MITDB-115.parquet`

**Citations:**
- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH arrhythmia database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50. https://doi.org/10.1161/01.CIR.101.23.e215
- PhysioNet: https://doi.org/10.13026/C2F305

### SMD (2 datasets)
Server machine operation data from Alibaba's infrastructure monitoring.

- `SMD-machine-1-2.parquet`
- `SMD-machine-3-5.parquet`

**Citation:** Su, Y., Zhao, Y., Niu, C., et al. (2019). Robust anomaly detection on attribute-dependent data. https://doi.org/10.1145/3292500.3330672

**Repository:** https://github.com/NetManAIOps/OmniAnomaly

### SVDB (2 datasets)
Sensor data from building monitoring systems.

- `SVDB-820.parquet`
- `SVDB-827.parquet`

**Citations:**
- MIT Library: https://dspace.mit.edu/handle/1721.1/29206
- PhysioNet: https://doi.org/10.13026/C2V30W

### Exathlon (1 dataset)
Simulated time series with synthetic anomalies.

- `Exathlon-5_1_100000_63-64.parquet`

**Citation:** https://doi.org/10.14778/3476249.3476307

**Repository:** https://github.com/exathlonbenchmark/exathlon

### CoMuT / S2Gpp (1 dataset)
Synthetic data with correlation-based anomalies.

- `CoMuT-ts_0.parquet`

**Citation:** Wenig, P., Schmidl, B., & Papenbrock, T. (2024). Anomaly Detectors for Multivariate Time Series: The Proof of the Pudding is in the Eating. Proceedings of the International Conference on Data Engineering Workshops (ICDEW).

**Repository:** https://hpi.de/naumann/s/comut

## License
These datasets are used for research and benchmarking purposes. Please refer to the original sources for specific licensing terms.
