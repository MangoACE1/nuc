from __future__ import annotations

import ast
import py_compile
from pathlib import Path

import pytest


PREDICT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_config_loads_the_fixed_deployment_contract() -> None:
    from netcatch_predict.config import load_config

    cfg = load_config(PREDICT_ROOT / "config.json")

    assert cfg.object_name == "obj1"
    assert cfg.payload_rigid_body == "p11"
    assert cfg.pose_topic == "/vrpn/obj1/pose"
    assert cfg.twist_topic == "/vrpn/obj1/twist"
    assert cfg.payload_pose_topic == "/vrpn/p11/pose"
    assert cfg.output_topic == "/netcatch/dynamics/prediction"
    assert cfg.frame_id == "mocap_world_enu"
    assert cfg.initial_payload_target_m == (-0.6, 0.6, 0.4)
    assert cfg.payload_target_z_m == 0.4
    assert cfg.catch_net_offset_z_fallback == 0.433
    assert cfg.catch_net_offset_z_fallback_m == 0.433
    assert cfg.prediction_plane_z_m == 0.833
    assert cfg.release_mode == "armed_auto"
    assert cfg.active_tracking_budget_s == 5.0
    assert cfg.gravity_mps2 == 9.81
    assert cfg.drag_beta_m_inv == 0.04
    assert cfg.publish_rate_hz == 30.0
    assert cfg.pose_stale_s == 0.05
    assert cfg.twist_stale_s == 0.02
    assert cfg.pose_only_grace_s == 0.05
    assert cfg.lost_after_gap_s == 0.25
    assert cfg.recovery_good_samples == 1
    assert cfg.consistency_samples == 3
    assert cfg.consistency_spread_max_m == 0.1
    assert cfg.deadband_m == 0.01
    assert cfg.jump_reject_m == 0.15
    assert cfg.intercept_deadline_grace_s == 0.20
    assert cfg.workspace_xy_min_m == (-0.8, -0.8)
    assert cfg.workspace_xy_max_m == (0.8, 0.8)
    assert cfg.usable_net_radius_m == 0.0
    assert cfg.estimator_replay_history_samples == 32


def test_config_rejects_missing_required_fields(complete_config_mapping: dict[str, object]) -> None:
    from netcatch_predict.config import ConfigError, PredictConfig

    complete_config_mapping.pop("pose_topic")

    with pytest.raises(ConfigError, match="missing.*pose_topic"):
        PredictConfig.from_mapping(complete_config_mapping)


def test_config_rejects_unknown_or_variance_named_noise_fields(
    complete_config_mapping: dict[str, object],
) -> None:
    from netcatch_predict.config import ConfigError, PredictConfig

    complete_config_mapping["pose_position_variance_m2"] = complete_config_mapping.pop(
        "pose_position_std_m"
    )

    with pytest.raises(ConfigError, match="unknown.*variance|missing.*std"):
        PredictConfig.from_mapping(complete_config_mapping)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("drag_beta_m_inv", -0.01),
        ("publish_rate_hz", 0.0),
        ("recovery_good_samples", 0),
        ("estimator_replay_history_samples", 0),
        ("workspace_xy_min_m", [0.9, -0.8]),
        ("initial_payload_target_m", [-0.6, 0.6, 0.5]),
        ("pose_stale_s", float("nan")),
    ],
)
def test_config_rejects_values_that_break_runtime_invariants(
    complete_config_mapping: dict[str, object], field: str, bad_value: object
) -> None:
    from netcatch_predict.config import ConfigError, PredictConfig

    complete_config_mapping[field] = bad_value

    with pytest.raises(ConfigError):
        PredictConfig.from_mapping(complete_config_mapping)


def test_config_accepts_a_positive_fixed_drag_beta(
    complete_config_mapping: dict[str, object],
) -> None:
    from netcatch_predict.config import PredictConfig

    complete_config_mapping["drag_beta_m_inv"] = 0.12

    assert PredictConfig.from_mapping(complete_config_mapping).drag_beta_m_inv == 0.12


def test_ros_adapter_compiles_while_core_modules_have_no_ros_imports(tmp_path: Path) -> None:
    package = PREDICT_ROOT / "netcatch_predict"
    ros_roots = {"rclpy", "geometry_msgs", "std_msgs", "std_srvs"}
    for filename in (
        "config.py",
        "estimator.py",
        "ballistics.py",
        "target_mapper.py",
        "session.py",
        "schema.py",
    ):
        tree = ast.parse((package / filename).read_text(encoding="utf-8"), filename=filename)
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
        assert imported_roots.isdisjoint(ros_roots), f"{filename} imports ROS: {imported_roots & ros_roots}"

    py_compile.compile(
        str(package / "node.py"),
        cfile=str(tmp_path / "node.pyc"),
        doraise=True,
    )
