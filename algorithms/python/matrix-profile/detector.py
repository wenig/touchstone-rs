"""Anomaly detector using matrix profiles via STUMPY.

This module implements a detector that computes matrix profiles on streaming
time series data to identify anomalies based on pattern recurrence.
"""
from collections import deque

import numpy as np
import stumpy
from touchstone_py import Detector, run_cli

_M = 100
_MAX_BUFFER = 3000


class Stumpi(Detector):
	"""Streaming anomaly detector based on matrix profiles.

	Uses the STUMPY library to compute matrix profiles on a sliding window
	of data. The anomaly score is the minimum distance in the matrix profile,
	aggregated across all dimensions.
	"""
    @classmethod
    def name(cls) -> str:
        """Return the name of this detector."""
        return "Stumpi"

    def __init__(self, n_dimensions: int) -> None:
        """Initialize the detector with the number of dimensions.

        Args:
            n_dimensions: Number of dimensions in the time series data.
        """
        self._n_dimensions = n_dimensions
        self._buffer = deque(maxlen=_MAX_BUFFER)

    def update(self, point: list[float]) -> float:
        """Update detector with a new data point and return anomaly score.

        Args:
            point: A list of float values, one per dimension.

        Returns:
            The anomaly score (sum of minimum distances across dimensions),
            or NaN if insufficient data has been accumulated (buffer size <= M).
        """
        self._buffer.append(point)
        if len(self._buffer) <= _M:
            return np.nan

        buf = np.array(self._buffer)
        Q = buf[-_M:]
        T = buf[:-1]

        score = 0.0
        for i in range(self._n_dimensions):
            dist_profile = stumpy.mass(Q[:, i], T[:, i])
            score += dist_profile.min()

        return score


if __name__ == "__main__":
    run_cli(Stumpi)
