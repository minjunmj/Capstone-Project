"""Minimal run logger: writes to both stdout and a run-local logs.txt."""

import logging
import sys
from pathlib import Path


def get_run_logger(run_dir: Path, name: str = "vdr") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "logs.txt")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
