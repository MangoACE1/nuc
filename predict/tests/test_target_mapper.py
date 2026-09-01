from __future__ import annotations

import numpy as np
import pytest


def _mapper():
    from netcatch_predict.target_mapper import TargetMapper

    return TargetMapper(
        initial_payload_target_m=[-0.6, 0.6, 0.4],
        payload_target_z_m=0.4,
        prediction_plane_z_m=0.833,
        consistency_samples=5,
        consistency_spread_max_m=0.05,
        deadband_m=0.01,
        jump_reject_m=0.15,
        workspace_xy_min_m=[-0.8, -0.8],
        workspace_xy_max_m=[0.8, 0.8],
        usable_net_radius_m=0.0,
    )


def _commit(mapper, xy: tuple[float, float] = (0.1, 0.2)) -> np.ndarray:
    result = None
    for _ in range(5):
        result = mapper.consider([xy[0], xy[1], 0.833])
    assert result is not None and result.committed
    return result.target_m.copy()


def test_initial_target_is_held_until_five_consistent_candidates_commit() -> None:
    mapper = _mapper()
    points = [
        [0.10, 0.20, 0.833],
        [0.11, 0.20, 0.833],
        [0.09, 0.20, 0.833],
        [0.10, 0.21, 0.833],
        [0.10, 0.19, 0.833],
    ]

    np.testing.assert_array_equal(mapper.current_target_m, [-0.6, 0.6, 0.4])
    for point in points[:4]:
        result = mapper.consider(point)
        assert result.accepted and not result.committed
        np.testing.assert_array_equal(result.target_m, [-0.6, 0.6, 0.4])

    result = mapper.consider(points[4])
    assert result.accepted and result.committed
    np.testing.assert_array_equal(result.target_m, [0.10, 0.19, 0.4])
    assert mapper.has_committed_target


def test_only_fifth_consistent_mapping_is_admitted_as_a_valid_prediction() -> None:
    from netcatch_predict.session import PredictionSession

    mapper = _mapper()
    session = PredictionSession(
        active_tracking_budget_s=5.0,
        recovery_good_samples=5,
        intercept_deadline_grace_s=0.2,
        initial_hold_target_m=[-0.6, 0.6, 0.4],
        payload_target_z_m=0.4,
    )
    session.release(monotonic_s=0.0)

    for index in range(1, 5):
        mapping = mapper.consider([0.1, 0.2, 0.833])
        admission = session.admit_prediction(
            command_ready=mapping.command_ready,
            monotonic_s=float(index),
            time_to_contact_s=10.0,
        )
        assert not admission.accepted
        assert not admission.valid
        assert session.deadline_monotonic_s is None
        assert session.active_elapsed_s(monotonic_s=float(index)) == 0.0
        np.testing.assert_array_equal(mapping.target_m, [-0.6, 0.6, 0.4])

    mapping = mapper.consider([0.1, 0.2, 0.833])
    admission = session.admit_prediction(
        command_ready=mapping.command_ready,
        monotonic_s=5.0,
        time_to_contact_s=10.0,
    )
    assert admission.accepted
    assert admission.valid
    assert session.deadline_monotonic_s == pytest.approx(15.2)
    np.testing.assert_array_equal(mapping.target_m, [0.1, 0.2, 0.4])


def test_zero_usable_radius_maps_committed_xy_exactly_to_intercept_xy() -> None:
    mapper = _mapper()

    target = _commit(mapper, xy=(0.123, -0.456))

    np.testing.assert_array_equal(target, [0.123, -0.456, 0.4])


def test_out_of_workspace_candidate_is_rejected_without_clipping() -> None:
    mapper = _mapper()

    result = mapper.consider([0.81, 0.0, 0.833])

    assert not result.accepted
    assert result.reason == "outside_workspace"
    assert mapper.pending_count == 0
    np.testing.assert_array_equal(result.target_m, [-0.6, 0.6, 0.4])


def test_one_large_candidate_jump_does_not_move_committed_target() -> None:
    mapper = _mapper()
    original = _commit(mapper)

    result = mapper.consider([0.4, 0.4, 0.833])

    assert not result.accepted
    assert result.reason == "candidate_jump"
    np.testing.assert_array_equal(result.target_m, original)
    np.testing.assert_array_equal(mapper.current_target_m, original)


def test_deadband_holds_target_for_small_candidate_motion() -> None:
    mapper = _mapper()
    original = _commit(mapper)

    result = mapper.consider([0.105, 0.2, 0.833])

    assert result.accepted and result.committed
    assert result.reason == "deadband_hold"
    np.testing.assert_array_equal(result.target_m, original)


def test_inconsistent_initial_window_does_not_commit() -> None:
    mapper = _mapper()
    for point in (
        [0.0, 0.0, 0.833],
        [0.01, 0.0, 0.833],
        [0.0, 0.01, 0.833],
        [0.01, 0.01, 0.833],
        [0.10, 0.10, 0.833],
    ):
        result = mapper.consider(point)

    assert result.accepted and not result.committed
    assert result.reason == "inconsistent_candidates"
    np.testing.assert_array_equal(mapper.current_target_m, [-0.6, 0.6, 0.4])


@pytest.mark.parametrize(
    "candidate",
    ([0.0, 0.0, 0.832], [0.0, np.nan, 0.833], [0.0, 0.833]),
)
def test_mapper_rejects_wrong_plane_nonfinite_and_malformed_candidates(candidate) -> None:
    mapper = _mapper()

    result = mapper.consider(candidate)

    assert not result.accepted
    np.testing.assert_array_equal(result.target_m, [-0.6, 0.6, 0.4])


def test_rearm_reset_restores_initial_hold_and_stability_window() -> None:
    mapper = _mapper()
    _commit(mapper)

    mapper.reset()

    assert not mapper.has_committed_target
    assert mapper.pending_count == 0
    np.testing.assert_array_equal(mapper.current_target_m, [-0.6, 0.6, 0.4])
