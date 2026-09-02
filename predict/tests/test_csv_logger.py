from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pytest


def test_timestamped_path_inserts_stamp_before_extension() -> None:
    from netcatch_predict.csv_logger import timestamped_csv_path

    now = datetime(2026, 9, 2, 10, 15, 0, 123456)
    path = timestamped_csv_path("logs/predict.csv", now=now)
    assert path == Path("logs/predict_20260902_101500_123456.csv")


def test_timestamped_path_treats_suffixless_path_as_directory() -> None:
    from netcatch_predict.csv_logger import timestamped_csv_path

    now = datetime(2026, 9, 2, 10, 15, 0, 123456)
    path = timestamped_csv_path("/tmp/predict_logs", now=now)
    assert path == Path("/tmp/predict_logs/predict_20260902_101500_123456.csv")


def test_logger_creates_parents_and_writes_header_and_rows(tmp_path: Path) -> None:
    from netcatch_predict.csv_logger import CSV_LOG_COLUMNS, CsvDebugLogger

    logger = CsvDebugLogger(tmp_path / "nested" / "debug.csv")
    try:
        assert logger.path.parent == tmp_path / "nested"
        assert logger.rows_written == 0
        row = {name: 0 for name in CSV_LOG_COLUMNS}
        row["state"] = "TRACKING"
        row["reason"] = "prediction_accepted"
        row["valid"] = 1
        logger.row(row)
        logger.row(row)
        assert logger.rows_written == 2
        with logger.path.open("r", encoding="utf-8", newline="") as stream:
            lines = list(csv.reader(stream))
        assert lines[0] == list(CSV_LOG_COLUMNS)
        assert len(lines) == 3
        assert lines[1][lines[0].index("state")] == "TRACKING"
        assert lines[1][lines[0].index("reason")] == "prediction_accepted"
    finally:
        logger.close()


def test_logger_requires_every_declared_column(tmp_path: Path) -> None:
    from netcatch_predict.csv_logger import CsvDebugLogger

    logger = CsvDebugLogger(tmp_path / "debug.csv")
    try:
        with pytest.raises(KeyError):
            logger.row({"state": "TRACKING"})
    finally:
        logger.close()


def test_logger_close_is_idempotent(tmp_path: Path) -> None:
    from netcatch_predict.csv_logger import CsvDebugLogger

    logger = CsvDebugLogger(tmp_path / "debug.csv")
    logger.close()
    logger.close()
