"""CSV debug logging for the prediction process.

Writes one row per published packet (30 Hz) once the EKF epoch has
started.  The output file is a timestamped sibling of the configured
path, e.g. ``predict_20260902_101500_123456.csv`` next to ``predict.csv``,
so each process run produces a fresh file and never overwrites history.

This module has no ROS imports and no dependency on the prediction core;
the node assembles and passes complete row dictionaries.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


CSV_LOG_COLUMNS: tuple[str, ...] = (
    "t_monotonic_s",
    "ekf_elapsed_s",
    "throw_id",
    "state",
    "valid",
    "reason",
    "pose_measured_x",
    "pose_measured_y",
    "pose_measured_z",
    "twist_measured_x",
    "twist_measured_y",
    "twist_measured_z",
    "ekf_x",
    "ekf_y",
    "ekf_z",
    "ekf_vx",
    "ekf_vy",
    "ekf_vz",
    "intercept_x",
    "intercept_y",
    "intercept_z",
    "time_to_contact_s",
    "intercept_time_ns",
    "xy_radius_95_m",
    "confidence",
    "command_target_x",
    "command_target_y",
    "command_target_z",
    "hold_target_x",
    "hold_target_y",
    "hold_target_z",
    "actual_capture_x",
    "actual_capture_y",
    "actual_capture_z",
    "actual_capture_time_ns",
    "state_time_ns",
)


def timestamped_csv_path(base_path: str | Path, *, now: datetime | None = None) -> Path:
    """Insert a start-time stamp into ``base_path``.

    ``logs/predict.csv`` becomes ``logs/predict_20260902_101500_123456.csv``;
    a suffix-less path is treated as a directory and gets ``predict_<stamp>.csv``
    appended inside it.
    """
    base = Path(base_path)
    stamp = (now if now is not None else datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    if base.suffix:
        return base.with_name(f"{base.stem}_{stamp}{base.suffix}")
    return base / f"predict_{stamp}.csv"


class CsvDebugLogger:
    """Append-only CSV writer with an exact declared column set."""

    def __init__(self, base_path: str | Path) -> None:
        self.path = timestamped_csv_path(base_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_LOG_COLUMNS)
        self.rows_written = 0

    def row(self, values: Mapping[str, Any]) -> None:
        """Append one row; every declared column must be present."""
        missing = [name for name in CSV_LOG_COLUMNS if name not in values]
        if missing:
            raise KeyError(f"csv row is missing columns: {', '.join(missing)}")
        self._writer.writerow([values[name] for name in CSV_LOG_COLUMNS])
        self.rows_written += 1
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
