"""Config loading and run-output directory helpers."""

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def make_run_dir(output_dir: str, experiment_name: str) -> Path:
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / "runs" / f"{timestamp}_{experiment_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(run_dir: Path, config: dict[str, Any]) -> None:
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
