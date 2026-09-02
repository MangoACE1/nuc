"""Pure monotonic-time lifecycle for one prediction throw."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class SessionState(str, Enum):
    WAIT_RELEASE = "WAIT_RELEASE"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RECOVERING = "RECOVERING"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class SessionTransition:
    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PredictionAdmission:
    accepted: bool
    valid: bool
    reason: str


class PredictionSession:
    def __init__(
        self,
        *,
        active_tracking_budget_s: float,
        recovery_good_samples: int,
        intercept_deadline_grace_s: float,
        initial_hold_target_m: Sequence[float],
        payload_target_z_m: float,
    ) -> None:
        initial_target = np.asarray(initial_hold_target_m, dtype=float)
        if initial_target.shape != (3,) or not np.all(np.isfinite(initial_target)):
            raise ValueError("initial hold target must be a finite three-vector")
        scalars = (active_tracking_budget_s, intercept_deadline_grace_s, payload_target_z_m)
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("session parameters must be finite")
        if active_tracking_budget_s <= 0.0 or intercept_deadline_grace_s < 0.0:
            raise ValueError("tracking budget must be positive and deadline grace nonnegative")
        if (
            isinstance(recovery_good_samples, bool)
            or not isinstance(recovery_good_samples, int)
            or recovery_good_samples <= 0
        ):
            raise ValueError("recovery_good_samples must be a positive integer")
        if not math.isclose(float(initial_target[2]), payload_target_z_m, abs_tol=1e-12):
            raise ValueError("initial hold z must equal payload target z")

        self._budget_s = float(active_tracking_budget_s)
        self._recovery_required = recovery_good_samples
        self._deadline_grace_s = float(intercept_deadline_grace_s)
        self._initial_hold_target = initial_target.copy()
        self._payload_target_z = float(payload_target_z_m)
        self.throw_id = 0
        self.filter_generation = 0
        self._last_monotonic_s: float | None = None
        self._reset_throw_state("waiting_for_arm")

    def _reset_throw_state(self, reason: str) -> None:
        self.state = SessionState.WAIT_RELEASE
        self.reason = reason
        self._armed = False
        self._budget_started = False
        self._active_accumulated_s = 0.0
        self._active_segment_start_s: float | None = None
        self.deadline_monotonic_s: float | None = None
        self.recovery_good_count = 0
        self._lost_hold_target: np.ndarray | None = None

    @property
    def can_accept_object_update(self) -> bool:
        return self.state not in (SessionState.DONE, SessionState.ERROR)

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def hold_target_m(self) -> np.ndarray:
        target = self._lost_hold_target if self._lost_hold_target is not None else self._initial_hold_target
        return target.copy()

    def _observe_time(self, monotonic_s: float) -> float:
        if not isinstance(monotonic_s, (int, float)) or not math.isfinite(monotonic_s):
            raise ValueError("monotonic_s must be finite")
        now = float(monotonic_s)
        if self._last_monotonic_s is not None and now < self._last_monotonic_s:
            raise ValueError("monotonic_s must not decrease")
        self._last_monotonic_s = now
        return now

    def _active_elapsed_raw(self, now: float) -> float:
        elapsed = self._active_accumulated_s
        if self._active_segment_start_s is not None:
            elapsed += now - self._active_segment_start_s
        return elapsed

    def active_elapsed_s(self, *, monotonic_s: float) -> float:
        now = self._observe_time(monotonic_s)
        self._apply_terminal_checks(now)
        return self._active_elapsed_raw(now)

    def _pause_active_segment(self, now: float) -> None:
        if self._active_segment_start_s is not None:
            self._active_accumulated_s += now - self._active_segment_start_s
            self._active_segment_start_s = None

    def _set_terminal(self, state: SessionState, reason: str, now: float) -> None:
        self._pause_active_segment(now)
        self.state = state
        self.reason = reason
        self.recovery_good_count = 0

    def _apply_terminal_checks(self, now: float) -> None:
        if self.state in (SessionState.DONE, SessionState.ERROR, SessionState.WAIT_RELEASE):
            return
        if self.deadline_monotonic_s is not None and now + 1e-12 >= self.deadline_monotonic_s:
            self._set_terminal(SessionState.DONE, "intercept_deadline_elapsed", now)
            return
        if self._budget_started and self._active_elapsed_raw(now) + 1e-12 >= self._budget_s:
            self._set_terminal(SessionState.DONE, "active_tracking_budget_exhausted", now)

    def tick(self, *, monotonic_s: float) -> SessionState:
        now = self._observe_time(monotonic_s)
        self._apply_terminal_checks(now)
        return self.state

    def release(self, *, monotonic_s: float) -> SessionTransition:
        now = self._observe_time(monotonic_s)
        self._apply_terminal_checks(now)
        if self.state is SessionState.DONE:
            return SessionTransition(False, "session_done")
        if self.state is SessionState.ERROR:
            return SessionTransition(False, "session_error")
        if self.state is not SessionState.WAIT_RELEASE:
            return SessionTransition(False, "already_released")
        self._armed = False
        self.state = SessionState.TRACKING
        self.reason = "released_waiting_for_prediction"
        return SessionTransition(True, self.reason)

    def arm(self, *, monotonic_s: float) -> SessionTransition:
        now = self._observe_time(monotonic_s)
        self._apply_terminal_checks(now)
        if self.state is SessionState.DONE:
            return SessionTransition(False, "session_done")
        if self.state is SessionState.ERROR:
            return SessionTransition(False, "session_error")
        if self.state is not SessionState.WAIT_RELEASE:
            return SessionTransition(False, "already_released")
        if self._armed:
            return SessionTransition(False, "already_armed")
        self._armed = True
        self.reason = "armed_waiting_for_free_flight"
        return SessionTransition(True, self.reason)

    def accept_prediction(
        self, *, monotonic_s: float, time_to_contact_s: float
    ) -> SessionTransition:
        if not isinstance(time_to_contact_s, (int, float)) or not math.isfinite(time_to_contact_s):
            raise ValueError("time_to_contact_s must be finite")
        if time_to_contact_s < 0.0:
            raise ValueError("time_to_contact_s must be nonnegative")
        now = self._observe_time(monotonic_s)
        self._apply_terminal_checks(now)
        if self.state is SessionState.DONE:
            return SessionTransition(False, "session_done")
        if self.state is SessionState.ERROR:
            return SessionTransition(False, "session_error")
        if self.state is SessionState.WAIT_RELEASE:
            return SessionTransition(False, "release_required")

        candidate_deadline = now + float(time_to_contact_s) + self._deadline_grace_s
        if self.deadline_monotonic_s is None:
            self.deadline_monotonic_s = candidate_deadline
        else:
            self.deadline_monotonic_s = min(self.deadline_monotonic_s, candidate_deadline)

        if not self._budget_started:
            self._budget_started = True

        if self.state is SessionState.TRACKING:
            if self._active_segment_start_s is None:
                self._active_segment_start_s = now
            self.reason = "prediction_accepted"
            return SessionTransition(True, self.reason)

        if self.state is SessionState.LOST:
            self.state = SessionState.RECOVERING
            self.recovery_good_count = 1
        else:
            self.recovery_good_count += 1
        if self.recovery_good_count >= self._recovery_required:
            self.state = SessionState.TRACKING
            self.recovery_good_count = 0
            if self._budget_started:
                self._active_segment_start_s = now
            self.reason = "recovered"
        else:
            self.reason = "recovering"
        return SessionTransition(True, self.reason)

    def admit_prediction(
        self,
        *,
        command_ready: bool,
        monotonic_s: float,
        time_to_contact_s: float,
    ) -> PredictionAdmission:
        """Gate a candidate before it can start timers or become publish-valid."""

        if type(command_ready) is not bool:
            raise ValueError("command_ready must be a boolean")
        if not command_ready:
            now = self._observe_time(monotonic_s)
            self._apply_terminal_checks(now)
            if self.state is SessionState.DONE:
                return PredictionAdmission(False, False, "session_done")
            if self.state is SessionState.ERROR:
                return PredictionAdmission(False, False, "session_error")
            return PredictionAdmission(False, False, "mapping_not_committed")
        transition = self.accept_prediction(
            monotonic_s=monotonic_s,
            time_to_contact_s=time_to_contact_s,
        )
        valid = transition.accepted and self.state is SessionState.TRACKING
        return PredictionAdmission(transition.accepted, valid, transition.reason)

    def mark_lost(
        self, *, monotonic_s: float, payload_position_m: Sequence[float]
    ) -> SessionTransition:
        payload = np.asarray(payload_position_m, dtype=float)
        if payload.shape != (3,) or not np.all(np.isfinite(payload)):
            raise ValueError("payload position must be a finite three-vector")
        now = self._observe_time(monotonic_s)
        self._apply_terminal_checks(now)
        if self.state is SessionState.DONE:
            return SessionTransition(False, "session_done")
        if self.state is SessionState.ERROR:
            return SessionTransition(False, "session_error")
        if self.state is SessionState.WAIT_RELEASE:
            return SessionTransition(False, "release_required")
        self._pause_active_segment(now)
        if self._lost_hold_target is None:
            self._lost_hold_target = np.array(
                [payload[0], payload[1], self._payload_target_z], dtype=float
            )
        self.state = SessionState.LOST
        self.recovery_good_count = 0
        self.reason = "tracking_lost"
        return SessionTransition(True, self.reason)

    def cancel(self, *, monotonic_s: float) -> SessionTransition:
        now = self._observe_time(monotonic_s)
        if self.state in (SessionState.DONE, SessionState.ERROR):
            return SessionTransition(False, "already_terminal")
        self._set_terminal(SessionState.DONE, "cancelled", now)
        return SessionTransition(True, self.reason)

    def finish(self, *, monotonic_s: float, reason: str) -> SessionTransition:
        """End the current throw immediately (e.g. on a fatal data gap)."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("finish reason must be a non-empty string")
        now = self._observe_time(monotonic_s)
        if self.state in (SessionState.DONE, SessionState.ERROR):
            return SessionTransition(False, "already_terminal")
        self._set_terminal(SessionState.DONE, reason, now)
        return SessionTransition(True, self.reason)

    def fail(self, *, monotonic_s: float, reason: str) -> SessionTransition:
        if not isinstance(reason, str) or not reason:
            raise ValueError("error reason must be a non-empty string")
        now = self._observe_time(monotonic_s)
        if self.state in (SessionState.DONE, SessionState.ERROR):
            return SessionTransition(False, "already_terminal")
        self._set_terminal(SessionState.ERROR, reason, now)
        return SessionTransition(True, self.reason)

    def rearm(self, *, monotonic_s: float) -> SessionTransition:
        self._observe_time(monotonic_s)
        self.throw_id += 1
        self.filter_generation += 1
        self._reset_throw_state("rearmed_waiting_for_arm")
        return SessionTransition(True, self.reason)
