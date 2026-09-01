from __future__ import annotations

import sys
from pathlib import Path

import pytest


PREDICT_ROOT = Path(__file__).resolve().parents[1]
if str(PREDICT_ROOT) not in sys.path:
    sys.path.insert(0, str(PREDICT_ROOT))


@pytest.fixture
def complete_config_mapping() -> dict[str, object]:
    return {
        "object_name": "obj1",
        "payload_rigid_body": "p11",
        "pose_topic": "/vrpn/obj1/pose",
        "twist_topic": "/vrpn/obj1/twist",
        "payload_pose_topic": "/vrpn/p11/pose",
        "output_topic": "/netcatch/dynamics/prediction",
        "frame_id": "mocap_world_enu",
        "initial_payload_target_m": [-0.6, 0.6, 0.4],
        "payload_target_z_m": 0.4,
        "catch_net_offset_z_fallback": 0.433,
        "prediction_plane_z_m": 0.833,
        "release_mode": "manual",
        "active_tracking_budget_s": 5.0,
        "gravity_mps2": 9.81,
        "drag_beta_m_inv": 0.0,
        "publish_rate_hz": 30.0,
        "pose_stale_s": 0.05,
        "twist_stale_s": 0.02,
        "pose_only_grace_s": 0.05,
        "recovery_good_samples": 5,
        "consistency_samples": 5,
        "consistency_spread_max_m": 0.05,
        "deadband_m": 0.01,
        "jump_reject_m": 0.15,
        "intercept_deadline_grace_s": 0.20,
        "workspace_xy_min_m": [-0.8, -0.8],
        "workspace_xy_max_m": [0.8, 0.8],
        "usable_net_radius_m": 0.0,
        "pose_position_std_m": 0.005,
        "twist_velocity_std_mps": 0.05,
        "process_acceleration_std_mps2": 2.0,
        "pose_innovation_gate_sigma": 5.0,
        "twist_innovation_gate_sigma": 5.0,
        "rk4_max_step_s": 0.005,
        "ballistic_max_time_s": 5.0,
        "estimator_replay_history_samples": 32,
    }
