from collections import deque

import numpy as np
import stumpy
from touchstone_py import Detector, run_cli

_M = 100
_MAX_BUFFER = 3000


class Stumpi(Detector):
    @classmethod
    def name(cls) -> str:
        return "Stumpi"

    def __init__(self, n_dimensions: int) -> None:
        self._n_dimensions = n_dimensions
        self._buffer = deque(maxlen=_MAX_BUFFER)

    def update(self, point: list[float]) -> float:
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
