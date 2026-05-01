# K-Means

A streaming K-Means anomaly detector that operates over sliding windows of multivariate observations.

## Parameters

| Symbol | Description |
|--------|-------------|
| `K` | Number of centroids (10) |
| `W` | Window length in time steps (50) |
| `η` | Online learning rate for centroid updates (0.007) |
| `warmup` | Points collected before initialization (149 = W + 100 − 1) |

## Model

Each centroid is a window of shape `(W, dim)`. The anomaly score for a new point is the Euclidean distance from the current sliding window to the nearest centroid; that centroid is then nudged toward the window at rate `η`:

```
score   = min_k dist(window, centroid_k)
centroid_k ← centroid_k + η · (window − centroid_k)
```

## Initialization

During the warmup phase the detector accumulates `WARMUP_LEN` points without scoring. Once enough points are collected, all overlapping windows of length `W` are extracted and centroids are seeded via **k-means++** (D²-weighted sampling), then refined with 20 Lloyd iterations. Empty clusters are reseeded from the window furthest from its current assignment.

Returns `NaN` during warmup and until the sliding window is full.
