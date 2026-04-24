import os

def run_touchstone(data_dir: os.PathLike, algorithms: list[type]) -> bytes:
    """Run Touchstone evaluation and return results as Arrow IPC bytes."""
