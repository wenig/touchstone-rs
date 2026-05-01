# Normal Distribution

A sliding-window z-score detector. It maintains the last 100 observations and scores each incoming point as the L2 norm of its per-dimension z-scores against that window:

```
score = sqrt( Σᵢ ((xᵢ − μᵢ) / σᵢ)² )
```

Running sums and sums-of-squares allow O(1) mean and variance updates as the window slides. Dimensions with zero variance contribute 0 to the norm. The score is computed *before* the point is added to the window, so a point is never compared against itself. Returns `NaN` until the window holds at least 2 observations.
