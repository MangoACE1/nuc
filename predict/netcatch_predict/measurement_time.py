"""Put asynchronous VRPN pose and twist samples on one NUC receive-time clock."""

from __future__ import annotations


def resolve_measurement_time_ns(
    *, stamp_sec: int, stamp_nanosec: int, receive_time_ns: int
) -> int:
    del stamp_sec, stamp_nanosec
    return receive_time_ns
