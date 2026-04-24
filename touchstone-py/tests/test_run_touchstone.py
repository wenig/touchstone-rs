"""Integration tests for touchstone_py.run_touchstone."""

import math
import sys
from pathlib import Path

import polars as pl
import pytest

from touchstone_py import Detector, run_touchstone, run_cli

EXPECTED_METRIC_COLS = {
    "ROC-AUC",
    "PR-AUC",
    "AvgPrec",
    "Precision",
    "Recall",
    "F1",
    "RangePrec",
    "RangeRec",
    "RangeF1",
    "RangePR-AUC",
    "RangePR-VUS",
    "RangeROC-VUS",
    "time_sec",
}


class ConstantDetector(Detector):
    """Returns a fixed score of 0.5 for every point."""

    def __init__(self, n_dimensions: int) -> None:
        pass

    @classmethod
    def name(cls) -> str:
        return "constant"

    def update(self, point: list[float]) -> float:
        return 0.5


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Temporary directory with one minimal labelled CSV dataset."""
    rows = ["timestamp,value,is_anomaly"]
    for i in range(50):
        label = 1 if i in {10, 20, 30} else 0
        rows.append(f"{i},{float(i)},{label}")
    (tmp_path / "dummy.csv").write_text("\n".join(rows))
    return tmp_path


def test_returns_polars_dataframe(data_dir: Path) -> None:
    result = run_touchstone(data_dir, [ConstantDetector])
    assert isinstance(result, pl.DataFrame)


def test_has_expected_columns(data_dir: Path) -> None:
    result = run_touchstone(data_dir, [ConstantDetector])
    assert {"dataset", "detector"} | EXPECTED_METRIC_COLS <= set(result.columns)


def test_one_row_per_dataset_detector(data_dir: Path) -> None:
    result = run_touchstone(data_dir, [ConstantDetector])
    assert result.height == 1
    assert result["dataset"][0] == "dummy"
    assert result["detector"][0] == "constant"


def test_time_sec_is_positive(data_dir: Path) -> None:
    result = run_touchstone(data_dir, [ConstantDetector])
    assert result["time_sec"][0] >= 0.0


def test_metric_scores_are_finite(data_dir: Path) -> None:
    result = run_touchstone(data_dir, [ConstantDetector])
    metric_cols = EXPECTED_METRIC_COLS - {"time_sec"}
    for col in metric_cols:
        value = result[col][0]
        assert value is not None and math.isfinite(value), f"{col} = {value}"


def test_multiple_detectors_one_row_each(data_dir: Path, tmp_path: Path) -> None:
    # Add a second dataset so the result table is 2 detectors × 2 datasets = 4 rows.
    rows = ["timestamp,value,is_anomaly"]
    for i in range(30):
        label = 1 if i == 15 else 0
        rows.append(f"{i},{float(i)},{label}")
    (tmp_path / "second.csv").write_text("\n".join(rows))

    # Reuse data_dir fixture manually via tmp_path — just copy the first dataset.
    first_csv = (data_dir / "dummy.csv").read_text()
    (tmp_path / "dummy.csv").write_text(first_csv)

    class ZeroDetector(Detector):
        def __init__(self, n_dimensions: int) -> None:
            pass

        @classmethod
        def name(cls) -> str:
            return "zero"

        def update(self, point: list[float]) -> float:
            return 0.0

    result = run_touchstone(tmp_path, [ConstantDetector, ZeroDetector])
    assert result.height == 4
    assert set(result["detector"].to_list()) == {"constant", "zero"}
    assert set(result["dataset"].to_list()) == {"dummy", "second"}


def test_warmup_nans_excluded(data_dir: Path) -> None:
    """A detector that returns NaN for the first half should still produce valid scores."""

    class WarmupDetector(Detector):
        def __init__(self, n_dimensions: int) -> None:
            self._count = 0

        @classmethod
        def name(cls) -> str:
            return "warmup"

        def update(self, point: list[float]) -> float:
            self._count += 1
            return float("nan") if self._count <= 25 else 0.5

    result = run_touchstone(data_dir, [WarmupDetector])
    assert result["ROC-AUC"][0] is not None


def test_run_cli_writes_csv(data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "--data-dir", str(data_dir)])
    run_cli(ConstantDetector)
    out = tmp_path / "touchstone-constant.csv"
    assert out.exists()
    df = pl.read_csv(out)
    assert df.height == 1
    assert "ROC-AUC" in df.columns
