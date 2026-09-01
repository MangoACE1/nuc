from __future__ import annotations

import json
import math
from typing import Any
from uuid import UUID


class ProtocolError(ValueError):
    """Raised when predictor JSON or a wire packet violates protocol v1."""


PREDICTION_FIELDS = frozenset(
    {
        "schema",
        "throw_id",
        "state",
        "valid",
        "frame_id",
        "object_name",
        "reason",
        "pose_time_ns",
        "twist_time_ns",
        "state_time_ns",
        "intercept_time_ns",
        "time_to_contact_s",
        "prediction_plane_z_m",
        "intercept_m",
        "command_target_m",
        "hold_target_m",
        "cov_xy_m2",
        "xy_radius_95_m",
        "confidence",
    }
)
ALLOWED_STATES = frozenset(
    {"WAIT_RELEASE", "TRACKING", "LOST", "RECOVERING", "DONE", "ERROR"}
)
WIRE_FIELDS = PREDICTION_FIELDS | frozenset(
    {"magic", "version", "source_boot_id", "seq", "send_time_ns"}
)


def _reject_constant(token: str) -> None:
    raise ProtocolError(f"non-finite JSON number is not allowed: {token}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json(payload: str | bytes) -> Any:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("payload is not valid UTF-8") from exc
    if not isinstance(payload, str):
        raise ProtocolError("payload must be str or bytes")
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from exc


def _require_nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolError(f"{name} must be a non-negative integer")
    return value


def _require_byte_budget(max_packet_bytes: Any) -> int:
    if type(max_packet_bytes) is not int or max_packet_bytes <= 0 or max_packet_bytes > 1200:
        raise ProtocolError("max_packet_bytes must be an integer in [1, 1200]")
    return max_packet_bytes


def _require_uuid4(value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError("source_boot_id must be a canonical UUID4 string")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("source_boot_id must be a canonical UUID4 string") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ProtocolError("source_boot_id must be a canonical UUID4 string")
    return value


def _require_finite_number(name: str, value: Any) -> float:
    if type(value) not in (int, float):
        raise ProtocolError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ProtocolError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ProtocolError(f"{name} must be a finite number")
    return normalized


def _seconds_to_nanoseconds(name: str, value: float) -> int:
    try:
        scaled = value * 1_000_000_000
    except (OverflowError, ValueError) as exc:
        raise ProtocolError(f"{name} cannot be represented in nanoseconds") from exc
    if not math.isfinite(scaled):
        raise ProtocolError(f"{name} cannot be represented in nanoseconds")
    try:
        return round(scaled)
    except (OverflowError, ValueError) as exc:
        raise ProtocolError(f"{name} cannot be represented in nanoseconds") from exc


def _require_vector3(name: str, value: Any) -> list[Any]:
    if not isinstance(value, list) or len(value) != 3:
        raise ProtocolError(f"{name} must be a three-element JSON array")
    for index, item in enumerate(value):
        _require_finite_number(f"{name}[{index}]", item)
    return value


def validate_prediction(
    value: Any,
    *,
    expected_frame_id: str,
    expected_object_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("prediction root must be a JSON object")
    actual_fields = set(value)
    if actual_fields != PREDICTION_FIELDS:
        missing = sorted(PREDICTION_FIELDS - actual_fields)
        extra = sorted(actual_fields - PREDICTION_FIELDS)
        raise ProtocolError(f"prediction fields mismatch; missing={missing}, extra={extra}")

    if value["schema"] != "netcatch.prediction.v1":
        raise ProtocolError("schema must be netcatch.prediction.v1")
    _require_nonnegative_int("throw_id", value["throw_id"])

    state = value["state"]
    if not isinstance(state, str) or state not in ALLOWED_STATES:
        raise ProtocolError(f"state must be one of {sorted(ALLOWED_STATES)}")
    if type(value["valid"]) is not bool:
        raise ProtocolError("valid must be a JSON boolean")
    if value["valid"] and state != "TRACKING":
        raise ProtocolError("valid=true is allowed only in TRACKING")

    if value["frame_id"] != expected_frame_id:
        raise ProtocolError("prediction frame_id does not match configuration")
    if value["object_name"] != expected_object_name:
        raise ProtocolError("prediction object_name does not match configuration")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ProtocolError("reason must be a non-empty string")

    pose_time_ns = _require_nonnegative_int("pose_time_ns", value["pose_time_ns"])
    twist_time_ns = _require_nonnegative_int("twist_time_ns", value["twist_time_ns"])
    state_time_ns = _require_nonnegative_int("state_time_ns", value["state_time_ns"])
    intercept_time_ns = _require_nonnegative_int(
        "intercept_time_ns", value["intercept_time_ns"]
    )
    if pose_time_ns > state_time_ns or twist_time_ns > state_time_ns:
        raise ProtocolError("pose_time_ns and twist_time_ns cannot exceed state_time_ns")

    time_to_contact = _require_finite_number(
        "time_to_contact_s", value["time_to_contact_s"]
    )
    if time_to_contact < 0.0:
        raise ProtocolError("time_to_contact_s must be non-negative")
    if value["valid"]:
        expected_intercept_time_ns = state_time_ns + _seconds_to_nanoseconds(
            "time_to_contact_s", time_to_contact
        )
        if intercept_time_ns < state_time_ns:
            raise ProtocolError("valid intercept_time_ns cannot precede state_time_ns")
        if abs(intercept_time_ns - expected_intercept_time_ns) > 1:
            raise ProtocolError(
                "valid intercept_time_ns must match state_time_ns + time_to_contact_s"
            )

    prediction_plane_z = _require_finite_number(
        "prediction_plane_z_m", value["prediction_plane_z_m"]
    )
    if abs(prediction_plane_z - 0.833) > 1e-6:
        raise ProtocolError("prediction_plane_z_m must equal 0.833 within 1e-6")

    intercept = _require_vector3("intercept_m", value["intercept_m"])
    command = _require_vector3("command_target_m", value["command_target_m"])
    hold = _require_vector3("hold_target_m", value["hold_target_m"])
    covariance = _require_vector3("cov_xy_m2", value["cov_xy_m2"])
    if abs(float(intercept[2]) - 0.833) > 1e-6:
        raise ProtocolError("intercept_m z must equal 0.833 within 1e-6")
    if abs(float(command[2]) - 0.4) > 1e-6:
        raise ProtocolError("command_target_m z must equal 0.4 within 1e-6")
    if abs(float(hold[2]) - 0.4) > 1e-6:
        raise ProtocolError("hold_target_m z must equal 0.4 within 1e-6")

    cov_xx, cov_xy, cov_yy = (float(item) for item in covariance)
    if cov_xx < 0.0 or cov_yy < 0.0 or cov_xx * cov_yy - cov_xy * cov_xy < -1e-12:
        raise ProtocolError("cov_xy_m2 must represent a positive-semidefinite covariance")

    xy_radius = _require_finite_number("xy_radius_95_m", value["xy_radius_95_m"])
    if xy_radius < 0.0:
        raise ProtocolError("xy_radius_95_m must be non-negative")
    confidence = _require_finite_number("confidence", value["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ProtocolError("confidence must be in [0, 1]")
    return dict(value)


def decode_prediction_json(
    payload: str | bytes,
    *,
    expected_frame_id: str,
    expected_object_name: str,
) -> dict[str, Any]:
    return validate_prediction(
        _load_strict_json(payload),
        expected_frame_id=expected_frame_id,
        expected_object_name=expected_object_name,
    )


def _validate_wire_object(
    value: Any,
    *,
    expected_frame_id: str,
    expected_object_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("wire packet root must be a JSON object")
    actual_fields = set(value)
    if actual_fields != WIRE_FIELDS:
        missing = sorted(WIRE_FIELDS - actual_fields)
        extra = sorted(actual_fields - WIRE_FIELDS)
        raise ProtocolError(f"wire fields mismatch; missing={missing}, extra={extra}")

    prediction = {field: value[field] for field in PREDICTION_FIELDS}
    validate_prediction(
        prediction,
        expected_frame_id=expected_frame_id,
        expected_object_name=expected_object_name,
    )
    if value["magic"] != "NETCATCH_DYNAMIC_TARGET":
        raise ProtocolError("magic must be NETCATCH_DYNAMIC_TARGET")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ProtocolError("version must be integer 1")
    _require_uuid4(value["source_boot_id"])
    _require_nonnegative_int("seq", value["seq"])
    send_time_ns = _require_nonnegative_int("send_time_ns", value["send_time_ns"])
    if send_time_ns == 0:
        raise ProtocolError("send_time_ns must be positive")
    return dict(value)


def encode_wire_packet(
    prediction: dict[str, Any],
    *,
    source_boot_id: str,
    seq: int,
    send_time_ns: int,
    max_packet_bytes: int,
    expected_frame_id: str,
    expected_object_name: str,
) -> bytes:
    budget = _require_byte_budget(max_packet_bytes)
    validated_prediction = validate_prediction(
        prediction,
        expected_frame_id=expected_frame_id,
        expected_object_name=expected_object_name,
    )
    packet = dict(validated_prediction)
    packet.update(
        {
            "magic": "NETCATCH_DYNAMIC_TARGET",
            "version": 1,
            "source_boot_id": source_boot_id,
            "seq": seq,
            "send_time_ns": send_time_ns,
        }
    )
    validated_packet = _validate_wire_object(
        packet,
        expected_frame_id=expected_frame_id,
        expected_object_name=expected_object_name,
    )
    encoded = json.dumps(
        validated_packet,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > budget:
        raise ProtocolError(f"packet too large: {len(encoded)} > {budget} bytes")
    return encoded


def decode_wire_packet(
    payload: str | bytes,
    *,
    expected_frame_id: str,
    expected_object_name: str,
    max_packet_bytes: int,
) -> dict[str, Any]:
    budget = _require_byte_budget(max_packet_bytes)
    if isinstance(payload, str):
        payload_size = len(payload.encode("utf-8"))
    elif isinstance(payload, bytes):
        payload_size = len(payload)
    else:
        raise ProtocolError("payload must be str or bytes")
    if payload_size > budget:
        raise ProtocolError(f"packet too large: {payload_size} > {budget} bytes")
    return _validate_wire_object(
        _load_strict_json(payload),
        expected_frame_id=expected_frame_id,
        expected_object_name=expected_object_name,
    )
