from __future__ import annotations

import time
from collections.abc import Callable
from uuid import uuid4

from .config import UdpConfig
from .protocol import decode_prediction_json, encode_wire_packet


class ErrorRecoveryGate:
    """Throttle error classes independently and signal one recovery per incident."""

    def __init__(
        self,
        *,
        interval_s: float = 1.0,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._interval_ns = round(interval_s * 1_000_000_000)
        self._monotonic_ns = monotonic_ns
        self._last_error_log_ns: dict[str, int] = {}

    def should_log_error(
        self,
        error_class: str,
        *,
        now_monotonic_ns: int | None = None,
    ) -> bool:
        now_ns = self._monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        last_ns = self._last_error_log_ns.get(error_class)
        if last_ns is None or now_ns - last_ns >= self._interval_ns:
            self._last_error_log_ns[error_class] = now_ns
            return True
        return False

    def record_success(self, error_class: str) -> bool:
        return self._last_error_log_ns.pop(error_class, None) is not None


class SenderCore:
    """ROS-independent prediction cache and datagram builder."""

    def __init__(
        self,
        config: UdpConfig,
        *,
        source_boot_id: str | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        time_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._config = config
        self._source_boot_id = source_boot_id if source_boot_id is not None else str(uuid4())
        self._monotonic_ns = monotonic_ns
        self._time_ns = time_ns
        self._stale_ns = int(config.prediction_stale_s * 1_000_000_000)
        self._latest_prediction = None
        self._received_monotonic_ns: int | None = None
        self._seq = 0

    def ingest_prediction(
        self,
        payload: str | bytes,
        *,
        received_monotonic_ns: int | None = None,
    ) -> None:
        prediction = decode_prediction_json(
            payload,
            expected_frame_id=self._config.frame_id,
            expected_object_name=self._config.object_name,
        )
        self._latest_prediction = prediction
        self._received_monotonic_ns = (
            self._monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
        )

    def next_packet(
        self,
        *,
        now_monotonic_ns: int | None = None,
        send_time_ns: int | None = None,
    ) -> bytes | None:
        if self._latest_prediction is None:
            return None
        if self._received_monotonic_ns is None:
            raise RuntimeError("prediction receipt timestamp is missing")
        if now_monotonic_ns is None:
            now_monotonic_ns = self._monotonic_ns()
        if send_time_ns is None:
            send_time_ns = self._time_ns()

        prediction = self._latest_prediction
        age_ns = max(0, now_monotonic_ns - self._received_monotonic_ns)
        if age_ns > self._stale_ns:
            prediction = dict(prediction)
            prediction.update(
                {
                    "state": "ERROR",
                    "valid": False,
                    "reason": "prediction_stale",
                    "time_to_contact_s": 0,
                }
            )
        packet = encode_wire_packet(
            prediction,
            source_boot_id=self._source_boot_id,
            seq=self._seq,
            send_time_ns=send_time_ns,
            max_packet_bytes=self._config.max_packet_bytes,
            expected_frame_id=self._config.frame_id,
            expected_object_name=self._config.object_name,
        )
        self._seq += 1
        return packet
