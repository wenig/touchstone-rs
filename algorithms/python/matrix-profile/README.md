# Matrix Profile

**Source:** [stumpy](https://stumpy.readthedocs.io) — Law et al., *STUMPY: A Powerful and Scalable Python Library for Time Series Data Mining*, JOSS 2019.

The matrix profile records, for each subsequence in a time series, the distance to its nearest non-overlapping neighbor. Low distances indicate recurring patterns; high distances indicate rare or anomalous subsequences. This detector maintains a sliding buffer of up to 3000 points and uses STUMPY's `mass` (Mueen's Algorithm for Similarity Search) to compute the distance profile of the most recent 100-point query against the rest of the buffer. The anomaly score is the minimum distance in that profile, summed across all dimensions. No score is produced until the buffer holds more than 100 points.
