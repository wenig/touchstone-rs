# Toto

**Source:** [Datadog/Toto-2.0-4m](https://huggingface.co/Datadog/Toto-2.0-4m) — Datadog, *Toto: Time Series Optimized Transformer for Observability*, 2025.

Toto is a pretrained time series forecasting model trained on a large corpus of observability metrics. It uses a transformer architecture with alternating time-wise and variate-wise attention blocks, enabling multivariate mixing across channels. This detector uses Toto zero-shot: given a 512-point context window, it forecasts the next 5 steps ahead in a single forward pass and scores each observation using the **continuous ranked probability score (CRPS)**, approximated via the mean pinball loss across all 9 quantile levels (0.1–0.9):

```
CRPS ≈ 2/9 · Σ_τ [ τ·max(y−q_τ, 0) + (1−τ)·max(q_τ−y, 0) ]
```

Tight quantiles that miss the actual value produce a higher score than wide quantiles that contain it, making CRPS sensitive to both forecast sharpness and calibration. Forecasting 5 steps per inference call reduces the number of model evaluations by 5× compared to single-step forecasting. The model produces no score during the initial 512-point warmup period.
