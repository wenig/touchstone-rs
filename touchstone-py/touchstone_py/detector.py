from abc import ABC, abstractmethod


class Detector(ABC):
    """Base class for streaming anomaly detectors evaluated by Touchstone.

    Subclasses must implement three members:

    - ``name()`` — classmethod returning a unique display name for results tables.
    - ``__init__(n_dimensions)`` — constructor called once per dataset; sets up
      internal state for a stream of that dimensionality.
    - ``update(point)`` — called once per data point in arrival order; returns an
      anomaly score (higher = more anomalous). Return ``float('nan')`` during any
      warm-up period — NaN scores are excluded from metric computation.
    """

    @abstractmethod
    def __init__(self, n_dimensions: int) -> None:
        """Initialise detector state for a stream with *n_dimensions* features."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return the detector's display name used in results tables."""

    @abstractmethod
    def update(self, point: list[float]) -> float:
        """Consume the next data point and return an anomaly score.

        Args:
            point: Feature vector of length *n_dimensions*.

        Returns:
            Anomaly score. Higher values indicate greater anomalousness.
            Return ``float('nan')`` during warm-up to exclude the point from
            metric computation.
        """
