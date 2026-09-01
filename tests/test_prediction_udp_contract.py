from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


NUC_DIR = Path(__file__).resolve().parents[1]
PREDICT_DIR = NUC_DIR / "predict"
UDP_DIR = NUC_DIR / "udp"

for package_root in (PREDICT_DIR, UDP_DIR, NUC_DIR):
    sys.path.insert(0, str(package_root))

from netcatch_predict.schema import encode_prediction_packet
from netcatch_udp.protocol import decode_prediction_json


def _prediction_packet() -> dict[str, object]:
    return {
        "schema": "netcatch.prediction.v1",
        "throw_id": 3,
        "state": "TRACKING",
        "valid": True,
        "frame_id": "mocap_world_enu",
        "object_name": "obj1",
        "reason": "prediction_accepted",
        "pose_time_ns": 990_000_000,
        "twist_time_ns": 995_000_000,
        "state_time_ns": 1_000_000_000,
        "intercept_time_ns": 1_250_000_000,
        "time_to_contact_s": 0.25,
        "prediction_plane_z_m": 0.833,
        "intercept_m": [-0.55, 0.55, 0.833],
        "command_target_m": [-0.6, 0.6, 0.4],
        "hold_target_m": [-0.6, 0.6, 0.4],
        "cov_xy_m2": [0.0004, 0.0, 0.0004],
        "xy_radius_95_m": 0.049,
        "confidence": 0.8,
    }


def _copy_configs(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    predict_values = json.loads((PREDICT_DIR / "config.json").read_text(encoding="utf-8"))
    udp_values = json.loads((UDP_DIR / "config.json").read_text(encoding="utf-8"))
    predict_path = tmp_path / "predict.json"
    udp_path = tmp_path / "udp.json"
    predict_path.write_text(json.dumps(predict_values), encoding="utf-8")
    udp_path.write_text(json.dumps(udp_values), encoding="utf-8")
    return predict_path, udp_path, predict_values, udp_values


def test_verifier_accepts_repository_contract_from_any_working_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(NUC_DIR / "verify_contract.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "NUC contract OK" in completed.stdout
    assert "/netcatch/dynamics/prediction" in completed.stdout
    assert "239.255.42.99:15150" in completed.stdout


def test_prediction_schema_output_is_accepted_unchanged_by_udp_decoder() -> None:
    prediction = _prediction_packet()

    encoded = encode_prediction_packet(prediction)
    decoded = decode_prediction_json(
        encoded,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
    )

    assert decoded == prediction
    assert decoded["command_target_m"] == [-0.6, 0.6, 0.4]


@pytest.mark.parametrize(
    ("side", "field", "invalid_value", "message"),
    [
        ("predict", "object_name", "ball", "predictor object_name must be exactly 'obj1'"),
        ("predict", "payload_rigid_body", "payload", "payload_rigid_body must be exactly 'p11'"),
        ("predict", "output_topic", "/wrong", "output_topic must be exactly '/netcatch/dynamics/prediction'"),
        ("udp", "prediction_topic", "/other", "UDP prediction_topic must equal predictor output_topic"),
        ("udp", "frame_id", "other_frame", "predictor and UDP frame_id values must match"),
        ("udp", "object_name", "obj2", "predictor and UDP object_name values must match"),
        ("predict", "initial_payload_target_m", [-0.5, 0.6, 0.4], "initial_payload_target_m must be exactly [-0.6, 0.6, 0.4]"),
        ("predict", "payload_target_z_m", 0.41, "payload_target_z_m must be exactly 0.4"),
        ("predict", "catch_net_offset_z_fallback", 0.4, "catch_net_offset_z_fallback must be exactly 0.433"),
        ("predict", "prediction_plane_z_m", 0.84, "prediction_plane_z_m must be exactly 0.833"),
        ("udp", "multicast_group", "239.1.2.3", "multicast_group must be exactly '239.255.42.99'"),
        ("udp", "multicast_port", 15151, "multicast_port must be exactly 15150"),
        ("udp", "multicast_ttl", 2, "multicast_ttl must be exactly 1"),
        ("udp", "publish_hz", 20.0, "publish_hz must be exactly 30 Hz"),
        ("udp", "prediction_stale_s", 0.2, "prediction_stale_s must be exactly 0.15 s"),
        ("udp", "max_packet_bytes", 1000, "max_packet_bytes must be exactly 1200"),
    ],
)
def test_invalid_cross_contract_values_fail_with_actionable_text(
    tmp_path: Path,
    side: str,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    from verify_contract import ContractError, verify_contract

    predict_path, udp_path, predict_values, udp_values = _copy_configs(tmp_path)
    values = predict_values if side == "predict" else udp_values
    values[field] = invalid_value
    path = predict_path if side == "predict" else udp_path
    path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ContractError, match=re.escape(message)):
        verify_contract(predict_config_path=predict_path, udp_config_path=udp_path)


def test_prediction_plane_must_equal_target_plus_fallback(tmp_path: Path) -> None:
    from verify_contract import ContractError, verify_contract

    predict_path, udp_path, predict_values, _ = _copy_configs(tmp_path)
    predict_values["payload_target_z_m"] = 0.4
    predict_values["catch_net_offset_z_fallback"] = 0.433
    predict_values["prediction_plane_z_m"] = 0.8330001
    predict_path.write_text(json.dumps(predict_values), encoding="utf-8")

    with pytest.raises(ContractError, match="prediction plane must equal payload target z plus fallback"):
        verify_contract(predict_config_path=predict_path, udp_config_path=udp_path)


@pytest.mark.parametrize(
    ("side", "field", "message"),
    [
        ("predict", "prediction_plane_z_m", "prediction_plane_z_m must be exactly 0.833"),
        ("udp", "publish_hz", "publish_hz must be exactly 30 Hz"),
    ],
)
def test_huge_json_numbers_are_actionable_contract_failures(
    tmp_path: Path,
    side: str,
    field: str,
    message: str,
) -> None:
    from verify_contract import ContractError, verify_contract

    predict_path, udp_path, predict_values, udp_values = _copy_configs(tmp_path)
    values = predict_values if side == "predict" else udp_values
    values[field] = 10**400
    path = predict_path if side == "predict" else udp_path
    path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ContractError, match=re.escape(message)):
        verify_contract(predict_config_path=predict_path, udp_config_path=udp_path)


def test_ast_audit_rejects_cross_package_and_ros_imports_outside_adapters(tmp_path: Path) -> None:
    from verify_contract import ContractError, audit_import_contract

    predict_package = tmp_path / "netcatch_predict"
    udp_package = tmp_path / "netcatch_udp"
    predict_package.mkdir()
    udp_package.mkdir()
    (predict_package / "node.py").write_text(
        "import rclpy\nfrom geometry_msgs.msg import PoseStamped\n",
        encoding="utf-8",
    )
    (udp_package / "sender_node.py").write_text(
        "import rclpy\nfrom std_msgs.msg import String\n",
        encoding="utf-8",
    )

    audit_import_contract(predict_package, udp_package)

    (predict_package / "core.py").write_text(
        "import netcatch_udp\nfrom rclpy.node import Node\n",
        encoding="utf-8",
    )
    (udp_package / "receiver_cli.py").write_text(
        "from netcatch_predict.schema import decode_prediction_packet\n"
        "from std_srvs.srv import Trigger\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError) as caught:
        audit_import_contract(predict_package, udp_package)

    message = str(caught.value)
    assert "predict/core.py imports forbidden package netcatch_udp" in message
    assert "predict/core.py imports ROS package rclpy" in message
    assert "udp/receiver_cli.py imports forbidden package netcatch_predict" in message
    assert "udp/receiver_cli.py imports ROS package std_srvs" in message
