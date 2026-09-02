from __future__ import annotations

import time

import numpy as np
import pytest


def _session():
    from netcatch_predict.session import PredictionSession

    return PredictionSession(
        active_tracking_budget_s=5.0,
        recovery_good_samples=5,
        intercept_deadline_grace_s=0.20,
        initial_hold_target_m=[-0.6, 0.6, 0.4],
        payload_target_z_m=0.4,
    )


def test_manual_release_does_not_start_budget_until_first_accepted_prediction() -> None:
    from netcatch_predict.session import SessionState

    session = _session()
    assert session.release(monotonic_s=100.0).accepted
    assert session.state is SessionState.TRACKING

    session.tick(monotonic_s=1_000.0)
    assert session.state is SessionState.TRACKING
    assert session.active_elapsed_s(monotonic_s=1_000.0) == 0.0

    assert session.accept_prediction(monotonic_s=1_000.0, time_to_contact_s=10.0).accepted
    session.tick(monotonic_s=1_004.999)
    assert session.state is SessionState.TRACKING
    session.tick(monotonic_s=1_005.0)
    assert session.state is SessionState.DONE
    assert session.reason == "active_tracking_budget_exhausted"


def test_active_budget_pauses_through_lost_and_recovering() -> None:
    from netcatch_predict.session import SessionState

    session = _session()
    session.release(monotonic_s=0.0)
    session.accept_prediction(monotonic_s=0.0, time_to_contact_s=100.0)
    session.mark_lost(monotonic_s=2.0, payload_position_m=[0.0, 0.0, 0.5])
    session.tick(monotonic_s=50.0)
    assert session.active_elapsed_s(monotonic_s=50.0) == pytest.approx(2.0)

    for sample_time in (50.0, 51.0, 52.0, 53.0):
        session.accept_prediction(monotonic_s=sample_time, time_to_contact_s=100.0)
        assert session.state is SessionState.RECOVERING
    session.accept_prediction(monotonic_s=54.0, time_to_contact_s=100.0)
    assert session.state is SessionState.TRACKING

    session.tick(monotonic_s=56.999)
    assert session.state is SessionState.TRACKING
    session.tick(monotonic_s=57.0)
    assert session.state is SessionState.DONE


def test_first_prediction_during_recovery_starts_budget_but_keeps_it_paused() -> None:
    from netcatch_predict.session import SessionState

    session = _session()
    session.release(monotonic_s=0.0)
    session.mark_lost(monotonic_s=1.0, payload_position_m=[0.0, 0.0, 0.5])

    for sample_time in (2.0, 3.0, 4.0, 5.0):
        session.accept_prediction(monotonic_s=sample_time, time_to_contact_s=100.0)
        assert session.active_elapsed_s(monotonic_s=sample_time) == 0.0
    session.accept_prediction(monotonic_s=6.0, time_to_contact_s=100.0)
    assert session.state is SessionState.TRACKING

    session.tick(monotonic_s=10.999)
    assert session.state is SessionState.TRACKING
    session.tick(monotonic_s=11.0)
    assert session.state is SessionState.DONE


def test_predicted_contact_deadline_does_not_pause_while_lost() -> None:
    from netcatch_predict.session import SessionState

    session = _session()
    session.release(monotonic_s=0.0)
    session.accept_prediction(monotonic_s=0.0, time_to_contact_s=3.0)
    assert session.deadline_monotonic_s == pytest.approx(3.2)
    session.mark_lost(monotonic_s=1.0, payload_position_m=[0.0, 0.0, 0.5])

    session.tick(monotonic_s=3.199)
    assert session.state is SessionState.LOST
    session.tick(monotonic_s=3.2)
    assert session.state is SessionState.DONE
    assert session.reason == "intercept_deadline_elapsed"


def test_lost_latches_payload_xy_once_and_forces_target_z() -> None:
    session = _session()
    session.release(monotonic_s=0.0)
    session.accept_prediction(monotonic_s=0.0, time_to_contact_s=100.0)

    session.mark_lost(monotonic_s=1.0, payload_position_m=[0.2, -0.3, 1.7])
    np.testing.assert_array_equal(session.hold_target_m, [0.2, -0.3, 0.4])
    session.mark_lost(monotonic_s=2.0, payload_position_m=[0.6, 0.7, -4.0])
    np.testing.assert_array_equal(session.hold_target_m, [0.2, -0.3, 0.4])


def test_recovery_needs_five_good_samples_without_changing_throw_or_filter_generation() -> None:
    from netcatch_predict.session import SessionState

    session = _session()
    session.release(monotonic_s=0.0)
    session.accept_prediction(monotonic_s=0.0, time_to_contact_s=100.0)
    session.mark_lost(monotonic_s=1.0, payload_position_m=[0.0, 0.0, 0.4])
    original_throw = session.throw_id
    original_filter_generation = session.filter_generation

    for index in range(4):
        session.accept_prediction(monotonic_s=2.0 + index, time_to_contact_s=100.0)
        assert session.state is SessionState.RECOVERING
    session.accept_prediction(monotonic_s=6.0, time_to_contact_s=100.0)

    assert session.state is SessionState.TRACKING
    assert session.throw_id == original_throw
    assert session.filter_generation == original_filter_generation
    assert session.recovery_good_count == 0


def test_done_rejects_updates_and_rearm_clears_all_per_throw_state() -> None:
    from netcatch_predict.session import SessionState

    session = _session()
    session.release(monotonic_s=0.0)
    session.accept_prediction(monotonic_s=0.0, time_to_contact_s=0.1)
    session.tick(monotonic_s=0.3)
    assert session.state is SessionState.DONE
    assert not session.can_accept_object_update
    rejected = session.accept_prediction(monotonic_s=0.31, time_to_contact_s=1.0)
    assert not rejected.accepted and rejected.reason == "session_done"

    session.rearm(monotonic_s=0.4)

    assert session.state is SessionState.WAIT_RELEASE
    assert session.throw_id == 1
    assert session.filter_generation == 1
    assert session.can_accept_object_update
    assert session.deadline_monotonic_s is None
    assert session.recovery_good_count == 0
    assert session.active_elapsed_s(monotonic_s=0.4) == 0.0
    np.testing.assert_array_equal(session.hold_target_m, [-0.6, 0.6, 0.4])


def test_wall_clock_jumps_do_not_change_monotonic_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from netcatch_predict.session import SessionState

    session = _session()
    session.release(monotonic_s=10.0)
    session.accept_prediction(monotonic_s=10.0, time_to_contact_s=100.0)

    monkeypatch.setattr(time, "time", lambda: -1e12)
    session.tick(monotonic_s=14.999)
    assert session.state is SessionState.TRACKING
    monkeypatch.setattr(time, "time", lambda: 1e12)
    session.tick(monotonic_s=15.0)
    assert session.state is SessionState.DONE


def test_cancel_and_error_are_terminal_until_rearm() -> None:
    from netcatch_predict.session import SessionState

    cancelled = _session()
    cancelled.cancel(monotonic_s=0.0)
    assert cancelled.state is SessionState.DONE
    assert not cancelled.can_accept_object_update

    failed = _session()
    failed.fail(monotonic_s=0.0, reason="estimator_failure")
    assert failed.state is SessionState.ERROR
    assert not failed.can_accept_object_update
    failed.rearm(monotonic_s=1.0)
    assert failed.state is SessionState.WAIT_RELEASE


def test_finish_ends_throw_immediately_until_rearm() -> None:
    from netcatch_predict.session import SessionState

    session = _session()
    session.release(monotonic_s=0.0)
    session.accept_prediction(monotonic_s=0.1, time_to_contact_s=0.5)

    transition = session.finish(monotonic_s=0.2, reason="pose_stale")
    assert transition.accepted
    assert session.state is SessionState.DONE
    assert session.reason == "pose_stale"
    assert not session.can_accept_object_update
    assert session.finish(monotonic_s=0.3, reason="again").accepted is False

    session.rearm(monotonic_s=0.4)
    assert session.state is SessionState.WAIT_RELEASE


def test_session_rejects_decreasing_or_nonfinite_monotonic_times() -> None:
    session = _session()
    session.release(monotonic_s=1.0)

    with pytest.raises(ValueError):
        session.tick(monotonic_s=0.9)
    with pytest.raises(ValueError):
        session.tick(monotonic_s=float("nan"))
