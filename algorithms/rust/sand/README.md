# SAND: Streaming Subsequence Anomaly Detection

Boniol et al., PVLDB 2021. doi:10.14778/3467861.3467863

## Parameters

| Symbol | Description |
|--------|-------------|
| `ℓ` | Target anomaly subsequence length |
| `ℓ_Θ` | Model subsequence length (`ℓ_Θ = 3 * ℓ` default) |
| `k` | Number of clusters per batch |
| `bsize` | Batch size |
| `α ∈ [0,1]` | Rate of change for weights and score normalization |

## Model

`Θ` is a set of tuples `(C̄_i, S_i, τ_i, w_i, s_i)` where:

- `C̄_i` — centroid (representative waveform of length `ℓ_Θ`)
- `S_i` — covariance matrix (`ℓ_Θ × ℓ_Θ`), accumulated outer products of cluster members; used to recompute the centroid without storing raw subsequences
- `τ_i` — intra-cluster average SBD distance; used as the merge threshold
- `w_i` — weight (normalized, sums to 1); reflects how frequently and recently this shape appeared
- `s_i` — cluster size at last update

## Distance: Shape-Based Distance (SBD)

SBD uses cross-correlation to find the best alignment between two sequences before measuring distance:

```
SBD(A, B) = 1 - max_w ( R_{w-ℓ}(A, B) / sqrt(R_0(A,A) * R_0(B,B)) )

R_k(A, B) = Σ A_{i+k} * B_i   for k ≥ 0
           = R_{-k}(B, A)       for k < 0
```

All subsequences are z-normalized before comparison.

## Initialization

1. Extract all overlapping subsequences of length `ℓ_Θ` from the first batch
2. Run k-Shape → get `k` clusters `{C_0, ..., C_k}`
3. For each cluster, compute:
   - Centroid `C̄_i` (eigenvector of largest eigenvalue of `Q^T S_i Q`)
   - `S_i = Σ_{T ∈ C_i} T * T^T`
   - Weight `w_i` proportional to `|C_i|^2 / Σ_j SBD(C̄_i, C̄_j)`
   - `τ_i = Σ_{T ∈ C_i} SBD(T, C̄_i)`

## Per-Batch Update

### 1. Cluster the new batch
Run k-Shape on the incoming batch → get `k` new clusters `C^t`.

### 2. Match and merge into Θ
For each new cluster `C_i^t`:

- If `∃ C̄_j ∈ Θ` such that `SBD(C̄_j, C̄_i^t) < τ_j`:
  - **Merge**: update `S_j ← S_j + Σ_{T ∈ C_i^t} T * T^T`, recompute `C̄_j`
  - Update threshold: weighted average of old `τ_j` and new cluster's intra-distance
- Else:
  - **Add** `C_i^t` as a new entry in `Θ`

### 3. Update weights
For each entry in `Θ`:

```
w_i^t = |C_i|^2 / Σ_j SBD(C̄_j, C̄_i)

A_i = t - t_last_i   (time since last batch that contributed to C_i)

w_i* = (1 - α) * w_i + α * w_i^t / max(1, A_i - bsize)
```

- `A_i ≤ bsize`: cluster is active this batch → no decay
- `A_i > bsize`: cluster is inactive → weight decays by `A_i - bsize`

Weights are normalized to sum to 1 after update.

## Anomaly Scoring

For each subsequence `T_{j,ℓ}` of length `ℓ` (shorter than `ℓ_Θ`) in the current batch:

```
d_j = Σ_{C̄_i ∈ Θ} w_i * min_{x ∈ [0, ℓ_Θ - ℓ]} dist(T_{j,ℓ}, (C̄_i)_{x,ℓ})
```

The `min_x` slides a window of length `ℓ` across each centroid to find the best-matching position (alignment).

Score is z-normalized using an exponential moving average of batch mean and std:

```
d_j* = (d_j - μ) / σ

μ* = α * μ(batch) + (1 - α) * μ
σ* = α * σ(batch) + (1 - α) * σ
```

## Complexity per Batch

```
O(max(
  bsize * k * ℓ_Θ * log(ℓ_Θ),   // k-Shape
  bsize * ℓ_Θ^2,                 // S matrix updates
  k * ℓ_Θ^3,                     // eigendecomposition
  |Θ| * k * ℓ_Θ * log(ℓ_Θ),     // matching
  bsize * ℓ_Θ * |Θ|              // scoring
))
```

Complexity is constant w.r.t. stream length (only `|Θ|` grows, slowly).
