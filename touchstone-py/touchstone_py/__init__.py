import argparse
import io
import os
from pathlib import Path

import polars as pl

from .detector import Detector
from ._rust_backend import run_touchstone as _run_touchstone_rust

__all__ = ["Detector", "run_touchstone", "run_cli"]


def run_touchstone(
    data_dir: os.PathLike,
    algorithms: list[type[Detector]],
) -> pl.DataFrame:
    """Run Touchstone evaluation against the given detector classes.

    Returns a polars DataFrame with columns: dataset, detector, <metrics>, time_sec.
    """
    ipc_bytes = _run_touchstone_rust(data_dir, algorithms)
    return pl.read_ipc(io.BytesIO(ipc_bytes))


def run_cli(detector: type[Detector]) -> None:
    """Parse ``--data-dir`` from the command line, evaluate *detector*, and write
    results to ``./touchstone-{name}.csv``.

    Mirrors the behaviour of the ``touchstone_main!`` Rust macro. Intended as the
    body of a ``__main__`` block in a detector module::

        if __name__ == "__main__":
            from touchstone_py import run_cli
            run_cli(MyDetector)
    """
    parser = argparse.ArgumentParser(description=f"Run Touchstone evaluation for {detector.name()}")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing Touchstone CSV datasets")
    args = parser.parse_args()

    df = run_touchstone(args.data_dir, [detector])
    out_path = Path(f"./touchstone-{detector.name()}.csv")
    df.write_csv(out_path)
    print(f"Results written to {out_path}")
