#!/usr/bin/env python3
"""Offline contract verification for the isolated NUC processes."""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


NUC_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICT_CONFIG = NUC_DIR / "predict" / "config.json"
DEFAULT_UDP_CONFIG = NUC_DIR / "udp" / "config.json"
DEFAULT_PREDICT_PACKAGE = NUC_DIR / "predict" / "netcatch_predict"
DEFAULT_UDP_PACKAGE = NUC_DIR / "udp" / "netcatch_udp"

ROS_IMPORT_ROOTS = frozenset(
    {
        "ament_index_python",
        "builtin_interfaces",
        "geometry_msgs",
        "launch",
        "launch_ros",
        "rclpy",
        "rosidl_runtime_py",
        "sensor_msgs",
        "std_msgs",
        "std_srvs",
        "tf2_ros",
    }
)


class ContractError(ValueError):
    """Raised when one or more NUC integration invariants are violated."""

    def __init__(self, errors: str | Iterable[str]) -> None:
        if isinstance(errors, str):
            normalized = (errors,)
        else:
            normalized = tuple(errors)
        self.errors = normalized
        super().__init__("\n".join(f"- {error}" for error in normalized))


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant {token} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load {label} config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} config root must be a JSON object: {path}")
    return value


def _is_ros_import(root: str) -> bool:
    return root in ROS_IMPORT_ROOTS or root.endswith(("_msgs", "_srvs", "_actions"))


def _import_roots(source_path: Path) -> set[str]:
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ContractError(f"cannot AST-audit {source_path}: {exc}") from exc

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def audit_import_contract(
    predict_package_dir: str | Path = DEFAULT_PREDICT_PACKAGE,
    udp_package_dir: str | Path = DEFAULT_UDP_PACKAGE,
) -> None:
    """Audit production imports without importing either production package."""

    package_specs = (
        ("predict", Path(predict_package_dir), "netcatch_udp", "node.py"),
        ("udp", Path(udp_package_dir), "netcatch_predict", "sender_node.py"),
    )
    errors: list[str] = []
    for label, package_dir, forbidden_package, ros_adapter in package_specs:
        source_paths = sorted(package_dir.rglob("*.py")) if package_dir.is_dir() else []
        if not source_paths:
            errors.append(f"{label} production package has no Python modules: {package_dir}")
            continue
        for source_path in source_paths:
            try:
                roots = _import_roots(source_path)
            except ContractError as exc:
                errors.extend(exc.errors)
                continue
            relative = source_path.relative_to(package_dir).as_posix()
            display = f"{label}/{relative}"
            if forbidden_package in roots:
                errors.append(f"{display} imports forbidden package {forbidden_package}")
            if relative != ros_adapter:
                for root in sorted(roots):
                    if _is_ros_import(root):
                        errors.append(f"{display} imports ROS package {root}")
    if errors:
        raise ContractError(errors)


def _finite_number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        return None
    return normalized if math.isfinite(normalized) else None


def _exact_number(value: object, expected: float) -> bool:
    normalized = _finite_number(value)
    return normalized is not None and normalized == expected


def verify_contract(
    *,
    predict_config_path: str | Path = DEFAULT_PREDICT_CONFIG,
    udp_config_path: str | Path = DEFAULT_UDP_CONFIG,
    predict_package_dir: str | Path = DEFAULT_PREDICT_PACKAGE,
    udp_package_dir: str | Path = DEFAULT_UDP_PACKAGE,
) -> None:
    """Validate config values and production-package isolation."""

    predict = _load_json_object(Path(predict_config_path), "predictor")
    udp = _load_json_object(Path(udp_config_path), "UDP")
    errors: list[str] = []

    if predict.get("object_name") != "obj1":
        errors.append("predictor object_name must be exactly 'obj1'")
    if predict.get("payload_rigid_body") != "p11":
        errors.append("payload_rigid_body must be exactly 'p11'")
    if predict.get("output_topic") != "/netcatch/dynamics/prediction":
        errors.append("output_topic must be exactly '/netcatch/dynamics/prediction'")
    if udp.get("prediction_topic") != predict.get("output_topic"):
        errors.append("UDP prediction_topic must equal predictor output_topic")
    if udp.get("frame_id") != predict.get("frame_id"):
        errors.append("predictor and UDP frame_id values must match")
    if udp.get("object_name") != predict.get("object_name"):
        errors.append("predictor and UDP object_name values must match")

    if predict.get("initial_payload_target_m") != [-0.6, 0.6, 0.4]:
        errors.append("initial_payload_target_m must be exactly [-0.6, 0.6, 0.4]")

    target_z = predict.get("payload_target_z_m")
    fallback_z = predict.get("catch_net_offset_z_fallback")
    plane_z = predict.get("prediction_plane_z_m")
    normalized_target_z = _finite_number(target_z)
    normalized_fallback_z = _finite_number(fallback_z)
    normalized_plane_z = _finite_number(plane_z)
    if not _exact_number(target_z, 0.4):
        errors.append("payload_target_z_m must be exactly 0.4")
    if not _exact_number(fallback_z, 0.433):
        errors.append("catch_net_offset_z_fallback must be exactly 0.433")
    if not _exact_number(plane_z, 0.833):
        errors.append("prediction_plane_z_m must be exactly 0.833")
    if not (
        normalized_target_z is not None
        and normalized_fallback_z is not None
        and normalized_plane_z is not None
        and math.isfinite(normalized_target_z + normalized_fallback_z)
        and normalized_plane_z == normalized_target_z + normalized_fallback_z
    ):
        errors.append("prediction plane must equal payload target z plus fallback")

    release_defaults: tuple[tuple[str, object, str], ...] = (
        ("release_mode", "armed_auto", "release_mode must be exactly 'armed_auto'"),
        ("release_buffer_s", 0.25, "release_buffer_s must be exactly 0.25 s"),
        ("release_fit_window_s", 0.05, "release_fit_window_s must be exactly 0.05 s"),
        ("release_min_pose_samples", 8, "release_min_pose_samples must be exactly 8"),
        ("release_min_twist_samples", 8, "release_min_twist_samples must be exactly 8"),
        ("release_min_speed_mps", 0.35, "release_min_speed_mps must be exactly 0.35 m/s"),
        (
            "release_max_position_rms_m",
            0.012,
            "release_max_position_rms_m must be exactly 0.012 m",
        ),
        (
            "release_max_velocity_rms_mps",
            0.20,
            "release_max_velocity_rms_mps must be exactly 0.20 m/s",
        ),
        (
            "release_max_acceleration_error_mps2",
            4.0,
            "release_max_acceleration_error_mps2 must be exactly 4.0 m/s^2",
        ),
        (
            "release_confirmation_windows",
            2,
            "release_confirmation_windows must be exactly 2",
        ),
    )
    for field, expected, message in release_defaults:
        value = predict.get(field)
        if type(expected) is float:
            matches = _exact_number(value, expected)
        else:
            matches = type(value) is type(expected) and value == expected
        if not matches:
            errors.append(message)

    udp_defaults: tuple[tuple[str, object, str], ...] = (
        ("multicast_group", "239.255.42.99", "multicast_group must be exactly '239.255.42.99'"),
        ("multicast_port", 15150, "multicast_port must be exactly 15150"),
        ("multicast_ttl", 1, "multicast_ttl must be exactly 1"),
        ("publish_hz", 30.0, "publish_hz must be exactly 30 Hz"),
        ("prediction_stale_s", 0.15, "prediction_stale_s must be exactly 0.15 s"),
        ("max_packet_bytes", 1200, "max_packet_bytes must be exactly 1200"),
    )
    for field, expected, message in udp_defaults:
        value = udp.get(field)
        if type(expected) is float:
            matches = _exact_number(value, expected)
        else:
            matches = type(value) is type(expected) and value == expected
        if not matches:
            errors.append(message)

    try:
        audit_import_contract(predict_package_dir, udp_package_dir)
    except ContractError as exc:
        errors.extend(exc.errors)
    if errors:
        raise ContractError(errors)


def main() -> int:
    try:
        verify_contract()
    except ContractError as exc:
        print(f"NUC contract FAILED\n{exc}", file=sys.stderr)
        return 1
    print(
        "NUC contract OK: obj1/p11; "
        "topic=/netcatch/dynamics/prediction; frame=mocap_world_enu; "
        "target=[-0.6,0.6,0.4]; plane=0.833; "
        "release=armed_auto window=0.05s confirm=2; "
        "UDP=239.255.42.99:15150 ttl=1 rate=30Hz stale=0.15s max=1200B; "
        "imports isolated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
