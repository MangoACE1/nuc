from __future__ import annotations

import json
import math
from dataclasses import dataclass
from ipaddress import AddressValueError, IPv4Address
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a deployment configuration violates the UDP contract."""


@dataclass(frozen=True)
class UdpConfig:
    prediction_topic: str
    multicast_group: str
    multicast_port: int
    multicast_ttl: int
    multicast_interface_ip: str
    publish_hz: float
    prediction_stale_s: float
    max_packet_bytes: int
    frame_id: str
    object_name: str


_CONFIG_FIELDS = frozenset(UdpConfig.__dataclass_fields__)


def _reject_constant(token: str) -> None:
    raise ConfigError(f"non-finite JSON number is not allowed: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _require_positive_finite(name: str, value: Any) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{name} must be a finite positive number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ConfigError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ConfigError(f"{name} must be a finite positive number")
    return normalized


def _validate(values: Any) -> UdpConfig:
    if not isinstance(values, dict):
        raise ConfigError("configuration root must be a JSON object")
    actual_fields = set(values)
    if actual_fields != _CONFIG_FIELDS:
        missing = sorted(_CONFIG_FIELDS - actual_fields)
        extra = sorted(actual_fields - _CONFIG_FIELDS)
        raise ConfigError(f"configuration fields mismatch; missing={missing}, extra={extra}")

    prediction_topic = values["prediction_topic"]
    if not isinstance(prediction_topic, str) or not prediction_topic.startswith("/"):
        raise ConfigError("prediction_topic must be an absolute ROS topic")

    multicast_group = values["multicast_group"]
    try:
        group_address = IPv4Address(multicast_group)
    except (AddressValueError, ValueError, TypeError) as exc:
        raise ConfigError("multicast_group must be an IPv4 multicast address") from exc
    if not group_address.is_multicast:
        raise ConfigError("multicast_group must be an IPv4 multicast address")

    interface_ip = values["multicast_interface_ip"]
    try:
        interface_address = IPv4Address(interface_ip)
    except (AddressValueError, ValueError, TypeError) as exc:
        raise ConfigError("multicast_interface_ip must be an IPv4 address") from exc
    if interface_address.is_multicast:
        raise ConfigError("multicast_interface_ip cannot be multicast")

    frame_id = values["frame_id"]
    object_name = values["object_name"]
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ConfigError("frame_id must be a non-empty string")
    if not isinstance(object_name, str) or not object_name.strip():
        raise ConfigError("object_name must be a non-empty string")

    return UdpConfig(
        prediction_topic=prediction_topic,
        multicast_group=str(group_address),
        multicast_port=_require_int("multicast_port", values["multicast_port"], 1, 65_535),
        multicast_ttl=_require_int("multicast_ttl", values["multicast_ttl"], 1, 255),
        multicast_interface_ip=str(interface_address),
        publish_hz=_require_positive_finite("publish_hz", values["publish_hz"]),
        prediction_stale_s=_require_positive_finite(
            "prediction_stale_s", values["prediction_stale_s"]
        ),
        max_packet_bytes=_require_int("max_packet_bytes", values["max_packet_bytes"], 1, 1200),
        frame_id=frame_id,
        object_name=object_name,
    )


def load_config(path: str | Path) -> UdpConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        values = json.load(
            stream,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    return _validate(values)
