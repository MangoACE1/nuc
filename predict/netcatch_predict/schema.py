"""Strict JSON schema helpers for ``netcatch.prediction.v1``."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence


PREDICTION_V1_KEYS = frozenset(
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
SESSION_STATES = frozenset(
    {"WAIT_RELEASE", "TRACKING", "LOST", "RECOVERING", "DONE", "ERROR"}
)
INVALID_STATES = frozenset({"WAIT_RELEASE", "LOST", "RECOVERING", "DONE", "ERROR"})


class SchemaError(ValueError):
    """Raised when a packet is not strict, finite prediction JSON v1."""


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{name} must be finite")
    return result


def _vector(name: str, value: object, length: int) -> list[float | int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise SchemaError(f"{name} must contain exactly {length} finite values")
    result = list(value)
    for index, item in enumerate(result):
        _number(f"{name}[{index}]", item)
    return result


def _timestamp(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a nonnegative integer")
    return value


def validate_prediction_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(packet, Mapping):
        raise SchemaError("packet must be a JSON object")
    supplied = set(packet)
    if supplied != PREDICTION_V1_KEYS:
        missing = sorted(PREDICTION_V1_KEYS - supplied)
        extra = sorted(supplied - PREDICTION_V1_KEYS)
        raise SchemaError(f"packet fields mismatch; missing={missing}, extra={extra}")

    result = dict(packet)
    if result["schema"] != "netcatch.prediction.v1":
        raise SchemaError("schema must be netcatch.prediction.v1")
    if isinstance(result["throw_id"], bool) or not isinstance(result["throw_id"], int):
        raise SchemaError("throw_id must be an integer")
    if result["throw_id"] < 0:
        raise SchemaError("throw_id must be nonnegative")
    if result["state"] not in SESSION_STATES:
        raise SchemaError("state is not a prediction session state")
    if type(result["valid"]) is not bool:
        raise SchemaError("valid must be a boolean")
    if result["state"] in INVALID_STATES and result["valid"]:
        raise SchemaError(f"{result['state']} packets must have valid=false")
    if result["valid"] and result["state"] != "TRACKING":
        raise SchemaError("only TRACKING packets may have valid=true")

    for name in ("frame_id", "object_name", "reason"):
        if not isinstance(result[name], str) or not result[name]:
            raise SchemaError(f"{name} must be a non-empty string")
    for name in (
        "pose_time_ns",
        "twist_time_ns",
        "state_time_ns",
        "intercept_time_ns",
    ):
        _timestamp(name, result[name])

    time_to_contact = _number("time_to_contact_s", result["time_to_contact_s"])
    plane = _number("prediction_plane_z_m", result["prediction_plane_z_m"])
    radius = _number("xy_radius_95_m", result["xy_radius_95_m"])
    confidence = _number("confidence", result["confidence"])
    if time_to_contact < 0.0:
        raise SchemaError("time_to_contact_s must be nonnegative")
    if radius < 0.0:
        raise SchemaError("xy_radius_95_m must be nonnegative")
    if not 0.0 <= confidence <= 1.0:
        raise SchemaError("confidence must lie in [0, 1]")
    if not math.isclose(plane, 0.833, abs_tol=1e-12):
        raise SchemaError("prediction_plane_z_m must be exactly 0.833")

    intercept = _vector("intercept_m", result["intercept_m"], 3)
    command_target = _vector("command_target_m", result["command_target_m"], 3)
    hold_target = _vector("hold_target_m", result["hold_target_m"], 3)
    covariance = _vector("cov_xy_m2", result["cov_xy_m2"], 3)
    if not math.isclose(float(intercept[2]), plane, abs_tol=1e-12):
        raise SchemaError("intercept z must equal the prediction plane")
    if not math.isclose(float(command_target[2]), 0.4, abs_tol=1e-12) or not math.isclose(
        float(hold_target[2]), 0.4, abs_tol=1e-12
    ):
        raise SchemaError("target z values must be exactly 0.4")
    pxx, pxy, pyy = map(float, covariance)
    if pxx < 0.0 or pyy < 0.0 or pxx * pyy - pxy * pxy < -1e-12:
        raise SchemaError("covariance must be positive semidefinite")

    if result["pose_time_ns"] > result["state_time_ns"]:
        raise SchemaError("pose_time_ns must not exceed state_time_ns")
    if result["twist_time_ns"] > result["state_time_ns"]:
        raise SchemaError("twist_time_ns must not exceed state_time_ns")
    if result["valid"]:
        if result["intercept_time_ns"] < result["state_time_ns"]:
            raise SchemaError("valid intercept_time_ns must be in the future")
        expected = result["state_time_ns"] + round(time_to_contact * 1e9)
        if abs(result["intercept_time_ns"] - expected) > 1:
            raise SchemaError("intercept_time_ns and time_to_contact_s disagree")
    return result


def encode_prediction_packet(packet: Mapping[str, Any]) -> str:
    validated = validate_prediction_packet(packet)
    try:
        return json.dumps(validated, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"packet is not finite JSON: {exc}") from exc


def _reject_constant(token: str) -> None:
    raise SchemaError(f"non-finite JSON constant {token} is forbidden")


def decode_prediction_packet(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise SchemaError("encoded packet must be a string")
    try:
        decoded = json.loads(raw, parse_constant=_reject_constant)
    except SchemaError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise SchemaError(f"invalid prediction JSON: {exc}") from exc
    return validate_prediction_packet(decoded)


def confidence_from_radius(xy_radius_95_m: float, usable_net_radius_m: float) -> float:
    radius = _number("xy_radius_95_m", xy_radius_95_m)
    usable = _number("usable_net_radius_m", usable_net_radius_m)
    if radius < 0.0 or usable < 0.0:
        raise SchemaError("radii must be nonnegative")
    if usable == 0.0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - radius / usable)))
