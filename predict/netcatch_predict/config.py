"""Validated configuration for the prediction process.

Noise inputs deliberately use standard-deviation names.  Consumers square
them once, at the point where a covariance is constructed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigError(ValueError):
    """Raised when a prediction configuration is incomplete or unsafe."""


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _vector(name: str, value: object, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{name} must contain {length} finite numbers")
    if len(value) != length:
        raise ConfigError(f"{name} must contain exactly {length} values")
    return tuple(_finite_float(f"{name}[{index}]", item) for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class PredictConfig:
    object_name: str
    payload_rigid_body: str
    pose_topic: str
    twist_topic: str
    payload_pose_topic: str
    output_topic: str
    frame_id: str
    initial_payload_target_m: tuple[float, float, float]
    payload_target_z_m: float
    catch_net_offset_z_fallback: float
    prediction_plane_z_m: float
    release_mode: str
    release_buffer_s: float
    release_fit_window_s: float
    release_min_pose_samples: int
    release_min_twist_samples: int
    release_min_speed_mps: float
    release_max_position_rms_m: float
    release_max_velocity_rms_mps: float
    release_max_acceleration_error_mps2: float
    release_confirmation_windows: int
    active_tracking_budget_s: float
    gravity_mps2: float
    drag_beta_m_inv: float
    publish_rate_hz: float
    pose_stale_s: float
    twist_stale_s: float
    pose_only_grace_s: float
    lost_after_gap_s: float
    recovery_good_samples: int
    consistency_samples: int
    consistency_spread_max_m: float
    deadband_m: float
    jump_reject_m: float
    intercept_deadline_grace_s: float
    workspace_xy_min_m: tuple[float, float]
    workspace_xy_max_m: tuple[float, float]
    usable_net_radius_m: float
    pose_position_std_m: float
    twist_velocity_std_mps: float
    process_acceleration_std_mps2: float
    pose_innovation_gate_sigma: float
    twist_innovation_gate_sigma: float
    rk4_max_step_s: float
    ballistic_max_time_s: float
    estimator_replay_history_samples: int

    @property
    def catch_net_offset_z_fallback_m(self) -> float:
        """Meter-unit alias for the deployment key named by the interface contract."""

        return self.catch_net_offset_z_fallback

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PredictConfig":
        if not isinstance(raw, Mapping):
            raise ConfigError("configuration root must be an object")

        expected = {field.name for field in fields(cls)}
        supplied = set(raw)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            parts: list[str] = []
            if unknown:
                parts.append(f"unknown config fields: {', '.join(unknown)}")
            if missing:
                parts.append(f"missing config fields: {', '.join(missing)}")
            raise ConfigError("; ".join(parts))

        string_fields = (
            "object_name",
            "payload_rigid_body",
            "pose_topic",
            "twist_topic",
            "payload_pose_topic",
            "output_topic",
            "frame_id",
            "release_mode",
        )
        strings: dict[str, str] = {}
        for name in string_fields:
            value = raw[name]
            if not isinstance(value, str) or not value:
                raise ConfigError(f"{name} must be a non-empty string")
            strings[name] = value

        for topic_name in ("pose_topic", "twist_topic", "payload_pose_topic", "output_topic"):
            if not strings[topic_name].startswith("/"):
                raise ConfigError(f"{topic_name} must be an absolute ROS topic")
        if strings["release_mode"] != "armed_auto":
            raise ConfigError("release_mode must be armed_auto")

        positive_float_names = (
            "active_tracking_budget_s",
            "release_buffer_s",
            "release_fit_window_s",
            "release_min_speed_mps",
            "release_max_position_rms_m",
            "release_max_velocity_rms_mps",
            "release_max_acceleration_error_mps2",
            "gravity_mps2",
            "publish_rate_hz",
            "pose_stale_s",
            "twist_stale_s",
            "lost_after_gap_s",
            "consistency_spread_max_m",
            "jump_reject_m",
            "pose_position_std_m",
            "twist_velocity_std_mps",
            "process_acceleration_std_mps2",
            "pose_innovation_gate_sigma",
            "twist_innovation_gate_sigma",
            "rk4_max_step_s",
            "ballistic_max_time_s",
        )
        numbers: dict[str, float] = {}
        for name in positive_float_names:
            numbers[name] = _finite_float(name, raw[name])
            if numbers[name] <= 0.0:
                raise ConfigError(f"{name} must be greater than zero")

        nonnegative_float_names = (
            "payload_target_z_m",
            "catch_net_offset_z_fallback",
            "prediction_plane_z_m",
            "drag_beta_m_inv",
            "pose_only_grace_s",
            "deadband_m",
            "intercept_deadline_grace_s",
            "usable_net_radius_m",
        )
        for name in nonnegative_float_names:
            numbers[name] = _finite_float(name, raw[name])
            if numbers[name] < 0.0:
                raise ConfigError(f"{name} must be nonnegative")

        integers = {
            "release_min_pose_samples": _positive_int(
                "release_min_pose_samples", raw["release_min_pose_samples"]
            ),
            "release_min_twist_samples": _positive_int(
                "release_min_twist_samples", raw["release_min_twist_samples"]
            ),
            "release_confirmation_windows": _positive_int(
                "release_confirmation_windows", raw["release_confirmation_windows"]
            ),
            "recovery_good_samples": _positive_int(
                "recovery_good_samples", raw["recovery_good_samples"]
            ),
            "consistency_samples": _positive_int("consistency_samples", raw["consistency_samples"]),
            "estimator_replay_history_samples": _positive_int(
                "estimator_replay_history_samples", raw["estimator_replay_history_samples"]
            ),
        }
        initial_target = _vector("initial_payload_target_m", raw["initial_payload_target_m"], 3)
        workspace_min = _vector("workspace_xy_min_m", raw["workspace_xy_min_m"], 2)
        workspace_max = _vector("workspace_xy_max_m", raw["workspace_xy_max_m"], 2)

        if any(low >= high for low, high in zip(workspace_min, workspace_max)):
            raise ConfigError("workspace_xy_min_m must be strictly below workspace_xy_max_m")
        if not all(low <= value <= high for value, low, high in zip(initial_target[:2], workspace_min, workspace_max)):
            raise ConfigError("initial_payload_target_m must lie inside the XY workspace")
        if not math.isclose(initial_target[2], numbers["payload_target_z_m"], abs_tol=1e-12):
            raise ConfigError("initial_payload_target_m z must equal payload_target_z_m")
        expected_plane = numbers["payload_target_z_m"] + numbers["catch_net_offset_z_fallback"]
        if not math.isclose(numbers["prediction_plane_z_m"], expected_plane, abs_tol=1e-12):
            raise ConfigError(
                "prediction_plane_z_m must equal payload_target_z_m plus catch_net_offset_z_fallback"
            )
        if numbers["prediction_plane_z_m"] <= numbers["payload_target_z_m"]:
            raise ConfigError("prediction plane must be above the payload target")
        if numbers["deadband_m"] > numbers["jump_reject_m"]:
            raise ConfigError("deadband_m must not exceed jump_reject_m")
        if numbers["release_buffer_s"] < numbers["release_fit_window_s"]:
            raise ConfigError(
                "release_buffer_s must be at least release_fit_window_s"
            )

        return cls(
            **strings,
            **numbers,
            **integers,
            initial_payload_target_m=initial_target,
            workspace_xy_min_m=workspace_min,
            workspace_xy_max_m=workspace_max,
        )


def load_config(path: str | Path) -> PredictConfig:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load configuration {config_path}: {exc}") from exc
    return PredictConfig.from_mapping(raw)
