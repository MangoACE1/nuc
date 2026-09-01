from __future__ import annotations

import numpy as np
import pytest


def _estimator(**overrides: float):
    from netcatch_predict.estimator import StateEstimator

    parameters = {
        "gravity_mps2": 9.81,
        "drag_beta_m_inv": 0.0,
        "pose_position_std_m": 0.01,
        "twist_velocity_std_mps": 0.05,
        "process_acceleration_std_mps2": 0.2,
        "pose_innovation_gate_sigma": 5.0,
        "twist_innovation_gate_sigma": 5.0,
        "rk4_max_step_s": 0.005,
        "estimator_replay_history_samples": 32,
    }
    parameters.update(overrides)
    return StateEstimator(**parameters)


def test_estimator_accepts_cross_source_asynchrony_and_replays_by_source_time() -> None:
    estimator = _estimator()

    twist_result = estimator.update_twist([1.0, 0.0, 2.0], time_ns=2_000_000_000)
    delayed_pose_result = estimator.update_pose([0.0, 0.0, 1.0], time_ns=1_000_000_000)
    snapshot = estimator.snapshot()

    assert twist_result.accepted
    assert delayed_pose_result.accepted
    assert snapshot.pose_time_ns == 1_000_000_000
    assert snapshot.twist_time_ns == 2_000_000_000
    assert snapshot.state_time_ns == 2_000_000_000
    assert snapshot.has_pose and snapshot.has_twist and snapshot.ready


@pytest.mark.parametrize("source", ["pose", "twist"])
def test_estimator_rejects_out_of_order_measurements_per_source(source: str) -> None:
    estimator = _estimator()
    update = estimator.update_pose if source == "pose" else estimator.update_twist

    assert update([0.0, 0.0, 0.0], time_ns=100).accepted
    result = update([0.0, 0.0, 0.0], time_ns=99)

    assert not result.accepted
    assert result.reason == f"out_of_order_{source}"
    assert estimator.measurement_counts[source] == 1


def test_pose_samples_do_not_create_a_native_velocity_measurement() -> None:
    estimator = _estimator(pose_innovation_gate_sigma=100.0)

    assert estimator.update_pose([0.0, 0.0, 1.0], time_ns=0).accepted
    assert estimator.update_pose([0.01, 0.0, 1.0], time_ns=10_000_000).accepted
    snapshot = estimator.snapshot()

    assert not snapshot.has_twist
    assert snapshot.twist_time_ns is None
    assert not snapshot.ready
    assert estimator.measurement_counts == {"pose": 2, "twist": 0}


def test_pose_and_twist_have_separate_innovation_gates() -> None:
    estimator = _estimator(
        pose_innovation_gate_sigma=0.5,
        twist_innovation_gate_sigma=10.0,
        twist_velocity_std_mps=0.1,
    )
    assert estimator.update_pose([0.0, 0.0, 1.0], time_ns=0).accepted
    assert estimator.update_twist([0.0, 0.0, 0.0], time_ns=0).accepted

    pose_result = estimator.update_pose([10.0, 0.0, 1.0], time_ns=1_000_000)
    twist_result = estimator.update_twist([0.1, 0.0, 0.0], time_ns=1_000_000)

    assert not pose_result.accepted
    assert pose_result.reason == "pose_innovation_gate"
    assert twist_result.accepted
    assert estimator.measurement_counts == {"pose": 1, "twist": 2}


def test_estimator_zero_drag_prediction_matches_constant_gravity_propagation() -> None:
    estimator = _estimator(
        pose_position_std_m=1e-6,
        twist_velocity_std_mps=1e-6,
        process_acceleration_std_mps2=1e-6,
        pose_innovation_gate_sigma=100.0,
        twist_innovation_gate_sigma=100.0,
    )
    assert estimator.update_pose([1.0, -2.0, 2.0], time_ns=0).accepted
    assert estimator.update_twist([0.5, 0.2, 1.0], time_ns=0).accepted

    predicted = estimator.estimate_at(500_000_000)

    np.testing.assert_allclose(
        predicted.state,
        [1.25, -1.9, 1.27375, 0.5, 0.2, -3.905],
        atol=2e-6,
    )
    assert predicted.state_time_ns == 500_000_000
    assert estimator.snapshot().state_time_ns == 0


def test_noise_standard_deviations_are_squared_once_when_covariances_are_built() -> None:
    estimator = _estimator(
        pose_position_std_m=0.2,
        twist_velocity_std_mps=0.3,
        process_acceleration_std_mps2=0.4,
    )

    np.testing.assert_allclose(estimator.pose_measurement_covariance, np.eye(3) * 0.04)
    np.testing.assert_allclose(estimator.twist_measurement_covariance, np.eye(3) * 0.09)
    assert estimator.process_acceleration_variance == pytest.approx(0.16)


def test_joseph_updates_keep_covariance_symmetric_and_positive_semidefinite() -> None:
    estimator = _estimator(pose_innovation_gate_sigma=100.0, twist_innovation_gate_sigma=100.0)
    for index in range(10):
        time_ns = index * 10_000_000
        assert estimator.update_pose([0.001 * index, 0.0, 1.0], time_ns=time_ns).accepted
        assert estimator.update_twist([0.1, 0.0, -0.01 * index], time_ns=time_ns).accepted

    covariance = estimator.snapshot().covariance
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_snapshot_arrays_are_copies_and_reset_clears_filter_state() -> None:
    estimator = _estimator()
    estimator.update_pose([0.0, 0.0, 1.0], time_ns=0)
    snapshot = estimator.snapshot()
    snapshot.state[:] = 99.0
    snapshot.covariance[:] = 99.0

    assert not np.all(estimator.snapshot().state == 99.0)
    estimator.reset()
    cleared = estimator.snapshot()
    assert cleared.state_time_ns is None
    assert cleared.pose_time_ns is None
    assert cleared.twist_time_ns is None
    assert not cleared.ready
    assert estimator.measurement_counts == {"pose": 0, "twist": 0}


@pytest.mark.parametrize(
    ("measurement", "time_ns"),
    [([0.0, np.nan, 0.0], 0), ([0.0, 0.0], 0), ([0.0, 0.0, 0.0], -1)],
)
def test_estimator_rejects_malformed_measurements(
    measurement: list[float], time_ns: int
) -> None:
    estimator = _estimator()

    with pytest.raises(ValueError):
        estimator.update_pose(measurement, time_ns=time_ns)


def test_globally_in_order_fast_path_never_replays_and_bounds_retained_history() -> None:
    estimator = _estimator(
        pose_innovation_gate_sigma=100.0,
        twist_innovation_gate_sigma=100.0,
        estimator_replay_history_samples=32,
    )

    for index in range(600):
        time_ns = index * 5_000_000
        update = estimator.update_pose if index % 2 == 0 else estimator.update_twist
        assert update([0.0, 0.0, 0.0], time_ns=time_ns).accepted

    assert estimator.retained_history_size <= 32
    assert estimator.replay_diagnostics == {
        "fast_path_updates": 600,
        "delayed_updates": 0,
        "replayed_records": 0,
    }


def test_delayed_cross_source_sample_inside_retained_window_is_replayed_and_accepted() -> None:
    estimator = _estimator(
        pose_innovation_gate_sigma=100.0,
        twist_innovation_gate_sigma=100.0,
        estimator_replay_history_samples=32,
    )
    assert estimator.update_pose([0.0, 0.0, 0.0], time_ns=0).accepted
    for time_ms in range(1, 21):
        assert estimator.update_twist([0.0, 0.0, 0.0], time_ns=time_ms * 1_000_000).accepted

    result = estimator.update_pose([0.0, 0.0, 0.0], time_ns=15_000_000)

    assert result.accepted
    assert estimator.snapshot().state_time_ns == 20_000_000
    assert estimator.replay_diagnostics["delayed_updates"] == 1
    assert estimator.replay_diagnostics["replayed_records"] == 5
    assert estimator.retained_history_size <= 32


def test_delayed_cross_source_sample_older_than_replay_base_is_rejected() -> None:
    estimator = _estimator(
        pose_innovation_gate_sigma=100.0,
        twist_innovation_gate_sigma=100.0,
        estimator_replay_history_samples=4,
    )
    assert estimator.update_pose([0.0, 0.0, 0.0], time_ns=0).accepted
    for time_ms in range(1, 11):
        assert estimator.update_twist([0.0, 0.0, 0.0], time_ns=time_ms * 1_000_000).accepted
    before = estimator.snapshot()

    result = estimator.update_pose([0.0, 0.0, 0.0], time_ns=5_000_000)

    assert not result.accepted
    assert result.reason == "delayed_beyond_replay_history"
    assert estimator.measurement_counts == {"pose": 1, "twist": 10}
    assert estimator.retained_history_size == 4
    np.testing.assert_array_equal(estimator.snapshot().state, before.state)
