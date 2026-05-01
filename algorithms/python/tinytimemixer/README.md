# TinyTimeMixer

**Source:** [ibm-granite/granite-timeseries-ttm-r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) — Ekambaram et al., *TTMs: Fast Multi-level Tiny Time Mixers for Improved Zero-shot and Few-shot Forecasting of Multivariate Time Series*, NeurIPS 2024.

TinyTimeMixer (TTM) is a compact pretrained forecasting model based on a hierarchical MLP-Mixer architecture with adaptive multi-resolution patching. It operates in a channel-independent mode, processing each time series dimension with shared weights across channels. This detector uses TTM zero-shot: given a 512-point context window, it predicts the next time step and treats the mean absolute error between the prediction and the actual observation as the anomaly score. The model produces no score during the initial 512-point warmup period.
