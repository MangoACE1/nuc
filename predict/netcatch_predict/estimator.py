"""Asynchronous six-state EKF using native pose and twist measurements."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .ballistics import propagate_state_and_transition


Source = Literal["pose", "twist"]


@dataclass(frozen=True, slots=True)
class UpdateResult:
    accepted: bool
    reason: str
    innovation_distance_squared: float | None = None


@dataclass(frozen=True, slots=True)
class EstimatorSnapshot:
    state: np.ndarray
    covariance: np.ndarray
    state_time_ns: int | None
    pose_time_ns: int | None
    twist_time_ns: int | None
    has_pose: bool
    has_twist: bool

    @property
    def ready(self) -> bool:
        return self.has_pose and self.has_twist and self.state_time_ns is not None


@dataclass(frozen=True, slots=True)
class _Measurement:
    time_ns: int
    sequence: int
    source: Source
    value: np.ndarray


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    state: np.ndarray
    covariance: np.ndarray
    time_ns: int


class StateEstimator:
    """Timestamp-ordered EKF with replay for delayed cross-source samples.

    Ordering is enforced independently for pose and twist.  A delayed pose may
    therefore arrive after a newer twist and is inserted at its source time;
    accepted later records are replayed without re-gating.
    """

    def __init__(
        self,
        *,
        gravity_mps2: float,
        drag_beta_m_inv: float,
        pose_position_std_m: float,
        twist_velocity_std_mps: float,
        process_acceleration_std_mps2: float,
        pose_innovation_gate_sigma: float,
        twist_innovation_gate_sigma: float,
        rk4_max_step_s: float = 0.005,
        estimator_replay_history_samples: int = 32,
    ) -> None:
        values = {
            "gravity_mps2": gravity_mps2,
            "drag_beta_m_inv": drag_beta_m_inv,
            "pose_position_std_m": pose_position_std_m,
            "twist_velocity_std_mps": twist_velocity_std_mps,
            "process_acceleration_std_mps2": process_acceleration_std_mps2,
            "pose_innovation_gate_sigma": pose_innovation_gate_sigma,
            "twist_innovation_gate_sigma": twist_innovation_gate_sigma,
            "rk4_max_step_s": rk4_max_step_s,
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("estimator parameters must be finite")
        if gravity_mps2 <= 0.0 or rk4_max_step_s <= 0.0:
            raise ValueError("gravity and RK4 max step must be positive")
        if drag_beta_m_inv < 0.0:
            raise ValueError("drag beta must be nonnegative")
        for name in (
            "pose_position_std_m",
            "twist_velocity_std_mps",
            "process_acceleration_std_mps2",
            "pose_innovation_gate_sigma",
            "twist_innovation_gate_sigma",
        ):
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(estimator_replay_history_samples, bool)
            or not isinstance(estimator_replay_history_samples, int)
            or estimator_replay_history_samples <= 0
        ):
            raise ValueError("estimator_replay_history_samples must be a positive integer")

        self.gravity_mps2 = float(gravity_mps2)
        self.drag_beta_m_inv = float(drag_beta_m_inv)
        self.rk4_max_step_s = float(rk4_max_step_s)
        self._pose_gate = float(pose_innovation_gate_sigma)
        self._twist_gate = float(twist_innovation_gate_sigma)
        self._pose_covariance = np.eye(3) * float(pose_position_std_m) ** 2
        self._twist_covariance = np.eye(3) * float(twist_velocity_std_mps) ** 2
        self.process_acceleration_variance = float(process_acceleration_std_mps2) ** 2
        self._history_capacity = estimator_replay_history_samples
        self._initial_covariance = np.eye(6) * 1e6
        self.reset()

    @property
    def pose_measurement_covariance(self) -> np.ndarray:
        return self._pose_covariance.copy()

    @property
    def twist_measurement_covariance(self) -> np.ndarray:
        return self._twist_covariance.copy()

    @property
    def measurement_counts(self) -> dict[str, int]:
        return dict(self._measurement_counts)

    @property
    def retained_history_size(self) -> int:
        return len(self._records)

    @property
    def replay_diagnostics(self) -> dict[str, int]:
        return dict(self._replay_diagnostics)

    def reset(self) -> None:
        self._records: list[_Measurement] = []
        self._checkpoints: list[_Checkpoint] = []
        self._next_sequence = 0
        self._latest_source_time: dict[Source, int | None] = {"pose": None, "twist": None}
        self._measurement_counts = {"pose": 0, "twist": 0}
        self._state = np.zeros(6, dtype=float)
        self._covariance = self._initial_covariance.copy()
        self._state_time_ns: int | None = None
        self._replay_base_state = self._state.copy()
        self._replay_base_covariance = self._covariance.copy()
        self._replay_base_time_ns: int | None = None
        self._replay_diagnostics = {
            "fast_path_updates": 0,
            "delayed_updates": 0,
            "replayed_records": 0,
        }

    def update_pose(self, position_m: Sequence[float], *, time_ns: int) -> UpdateResult:
        return self._add_measurement("pose", position_m, time_ns)

    def update_twist(self, velocity_mps: Sequence[float], *, time_ns: int) -> UpdateResult:
        return self._add_measurement("twist", velocity_mps, time_ns)

    def _add_measurement(
        self, source: Source, measurement: Sequence[float], time_ns: int
    ) -> UpdateResult:
        value = np.asarray(measurement, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{source} measurement must be a finite three-vector")
        if isinstance(time_ns, bool) or not isinstance(time_ns, int) or time_ns < 0:
            raise ValueError("measurement time_ns must be a nonnegative integer")
        latest = self._latest_source_time[source]
        if latest is not None and time_ns <= latest:
            return UpdateResult(False, f"out_of_order_{source}")

        candidate = _Measurement(time_ns, self._next_sequence, source, value.copy())
        if self._state_time_ns is None or time_ns >= self._state_time_ns:
            result = self._accept_fast_path(candidate)
        else:
            result = self._accept_delayed(candidate)
        if not result.accepted:
            return result

        self._latest_source_time[source] = time_ns
        self._measurement_counts[source] += 1
        self._next_sequence += 1
        self._trim_history()
        return result

    def _advance_to_record(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        state_time_ns: int | None,
        record: _Measurement,
        *,
        gate: bool,
    ) -> tuple[np.ndarray, np.ndarray, int, UpdateResult]:
        if state_time_ns is None:
            state_time_ns = record.time_ns
        elif record.time_ns > state_time_ns:
            state, covariance = self._propagate(
                state, covariance, (record.time_ns - state_time_ns) * 1e-9
            )
            state_time_ns = record.time_ns
        state, covariance, result = self._measurement_update(
            state, covariance, record, gate=gate
        )
        return state, covariance, state_time_ns, result

    def _accept_fast_path(self, candidate: _Measurement) -> UpdateResult:
        state, covariance, state_time_ns, result = self._advance_to_record(
            self._state.copy(),
            self._covariance.copy(),
            self._state_time_ns,
            candidate,
            gate=True,
        )
        if not result.accepted:
            return result
        self._records.append(candidate)
        self._checkpoints.append(
            _Checkpoint(state.copy(), covariance.copy(), state_time_ns)
        )
        self._state = state
        self._covariance = covariance
        self._state_time_ns = state_time_ns
        self._replay_diagnostics["fast_path_updates"] += 1
        return result

    def _accept_delayed(self, candidate: _Measurement) -> UpdateResult:
        if (
            self._replay_base_time_ns is not None
            and candidate.time_ns < self._replay_base_time_ns
        ):
            return UpdateResult(False, "delayed_beyond_replay_history")

        keys = [(record.time_ns, record.sequence) for record in self._records]
        insert_at = bisect_right(keys, (candidate.time_ns, candidate.sequence))
        if insert_at == 0:
            state = self._replay_base_state.copy()
            covariance = self._replay_base_covariance.copy()
            state_time_ns = self._replay_base_time_ns
        else:
            checkpoint = self._checkpoints[insert_at - 1]
            state = checkpoint.state.copy()
            covariance = checkpoint.covariance.copy()
            state_time_ns = checkpoint.time_ns

        state, covariance, state_time_ns, result = self._advance_to_record(
            state, covariance, state_time_ns, candidate, gate=True
        )
        if not result.accepted:
            return result

        records = [*self._records[:insert_at], candidate, *self._records[insert_at:]]
        checkpoints = list(self._checkpoints[:insert_at])
        checkpoints.append(_Checkpoint(state.copy(), covariance.copy(), state_time_ns))
        replayed_records = 0
        for record in self._records[insert_at:]:
            state, covariance, state_time_ns, _ = self._advance_to_record(
                state, covariance, state_time_ns, record, gate=False
            )
            checkpoints.append(_Checkpoint(state.copy(), covariance.copy(), state_time_ns))
            replayed_records += 1

        self._records = records
        self._checkpoints = checkpoints
        self._state = state
        self._covariance = covariance
        self._state_time_ns = state_time_ns
        self._replay_diagnostics["delayed_updates"] += 1
        self._replay_diagnostics["replayed_records"] += replayed_records
        return result

    def _trim_history(self) -> None:
        excess = len(self._records) - self._history_capacity
        if excess <= 0:
            return
        checkpoint = self._checkpoints[excess - 1]
        self._replay_base_state = checkpoint.state.copy()
        self._replay_base_covariance = checkpoint.covariance.copy()
        self._replay_base_time_ns = checkpoint.time_ns
        self._records = self._records[excess:]
        self._checkpoints = self._checkpoints[excess:]

    def _measurement_update(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        record: _Measurement,
        *,
        gate: bool,
    ) -> tuple[np.ndarray, np.ndarray, UpdateResult]:
        measurement_matrix = np.zeros((3, 6), dtype=float)
        if record.source == "pose":
            measurement_matrix[:, :3] = np.eye(3)
            measurement_covariance = self._pose_covariance
            gate_sigma = self._pose_gate
        else:
            measurement_matrix[:, 3:] = np.eye(3)
            measurement_covariance = self._twist_covariance
            gate_sigma = self._twist_gate

        innovation = record.value - measurement_matrix @ state
        innovation_covariance = (
            measurement_matrix @ covariance @ measurement_matrix.T + measurement_covariance
        )
        try:
            solved = np.linalg.solve(innovation_covariance, innovation)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(innovation_covariance) @ innovation
        distance_squared = float(innovation @ solved)
        if gate and distance_squared > gate_sigma * gate_sigma:
            return (
                state,
                covariance,
                UpdateResult(False, f"{record.source}_innovation_gate", distance_squared),
            )

        gain = np.linalg.solve(
            innovation_covariance, measurement_matrix @ covariance
        ).T
        updated_state = state + gain @ innovation
        identity_minus_kh = np.eye(6) - gain @ measurement_matrix
        updated_covariance = (
            identity_minus_kh @ covariance @ identity_minus_kh.T
            + gain @ measurement_covariance @ gain.T
        )
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        return updated_state, updated_covariance, UpdateResult(True, "accepted", distance_squared)

    def _propagate(
        self, state: np.ndarray, covariance: np.ndarray, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        propagated_state, transition = propagate_state_and_transition(
            state,
            duration_s,
            gravity_mps2=self.gravity_mps2,
            drag_beta_m_inv=self.drag_beta_m_inv,
            max_step_s=self.rk4_max_step_s,
        )
        identity3 = np.eye(3)
        process_covariance = self.process_acceleration_variance * np.block(
            [
                [identity3 * duration_s**4 / 4.0, identity3 * duration_s**3 / 2.0],
                [identity3 * duration_s**3 / 2.0, identity3 * duration_s**2],
            ]
        )
        propagated_covariance = transition @ covariance @ transition.T + process_covariance
        propagated_covariance = 0.5 * (propagated_covariance + propagated_covariance.T)
        return propagated_state, propagated_covariance

    def snapshot(self) -> EstimatorSnapshot:
        return EstimatorSnapshot(
            state=self._state.copy(),
            covariance=self._covariance.copy(),
            state_time_ns=self._state_time_ns,
            pose_time_ns=self._latest_source_time["pose"],
            twist_time_ns=self._latest_source_time["twist"],
            has_pose=self._latest_source_time["pose"] is not None,
            has_twist=self._latest_source_time["twist"] is not None,
        )

    def estimate_at(self, time_ns: int) -> EstimatorSnapshot:
        if isinstance(time_ns, bool) or not isinstance(time_ns, int) or time_ns < 0:
            raise ValueError("time_ns must be a nonnegative integer")
        if self._state_time_ns is None:
            raise RuntimeError("estimator has no accepted measurements")
        if time_ns < self._state_time_ns:
            raise ValueError("cannot estimate before the current filter state")
        state, covariance = self._propagate(
            self._state,
            self._covariance,
            (time_ns - self._state_time_ns) * 1e-9,
        )
        return EstimatorSnapshot(
            state=state,
            covariance=covariance,
            state_time_ns=time_ns,
            pose_time_ns=self._latest_source_time["pose"],
            twist_time_ns=self._latest_source_time["twist"],
            has_pose=self._latest_source_time["pose"] is not None,
            has_twist=self._latest_source_time["twist"] is not None,
        )
