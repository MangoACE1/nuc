"""Causal free-flight onset detection for an armed hand-thrown object."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


MeasurementSource = Literal["pose", "twist"]


@dataclass(frozen=True, slots=True)
class BufferedMeasurement:
    source: MeasurementSource
    time_ns: int
    receive_time_ns: int
    value: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class FreeFlightDetection:
    release_time_ns: int
    confirmation_time_ns: int
    position_rms_m: float
    velocity_rms_mps: float
    acceleration_error_mps2: float
    measurements: tuple[BufferedMeasurement, ...]


class ReleaseDetector:
    """Confirm a ballistic suffix, then backdate to its first measurement."""

    def __init__(
        self,
        *,
        gravity_mps2: float,
        buffer_duration_s: float,
        fit_window_s: float,
        min_pose_samples: int,
        min_twist_samples: int,
        min_speed_mps: float,
        max_position_rms_m: float,
        max_velocity_rms_mps: float,
        max_acceleration_error_mps2: float,
        confirmation_windows: int,
    ) -> None:
        positive_floats = {
            "gravity_mps2": gravity_mps2,
            "buffer_duration_s": buffer_duration_s,
            "fit_window_s": fit_window_s,
            "min_speed_mps": min_speed_mps,
            "max_position_rms_m": max_position_rms_m,
            "max_velocity_rms_mps": max_velocity_rms_mps,
            "max_acceleration_error_mps2": max_acceleration_error_mps2,
        }
        for name, value in positive_floats.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if buffer_duration_s < fit_window_s:
            raise ValueError("buffer_duration_s must be at least fit_window_s")
        for name, value in (
            ("min_pose_samples", min_pose_samples),
            ("min_twist_samples", min_twist_samples),
            ("confirmation_windows", confirmation_windows),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        self.gravity_mps2 = float(gravity_mps2)
        self.buffer_duration_ns = round(float(buffer_duration_s) * 1e9)
        self.fit_window_ns = round(float(fit_window_s) * 1e9)
        self.min_pose_samples = min_pose_samples
        self.min_twist_samples = min_twist_samples
        self.min_speed_mps = float(min_speed_mps)
        self.max_position_rms_m = float(max_position_rms_m)
        self.max_velocity_rms_mps = float(max_velocity_rms_mps)
        self.max_acceleration_error_mps2 = float(max_acceleration_error_mps2)
        self.confirmation_windows = confirmation_windows
        self.reset()

    def reset(self) -> None:
        self._measurements: list[BufferedMeasurement] = []
        self._latest_source_time_ns: dict[MeasurementSource, int | None] = {
            "pose": None,
            "twist": None,
        }
        self._consecutive_passes = 0
        self._candidate_release_time_ns: int | None = None
        self._detection: FreeFlightDetection | None = None

    @property
    def buffered_measurement_count(self) -> int:
        return len(self._measurements)

    def observe_pose(
        self,
        position_m: Sequence[float],
        *,
        time_ns: int,
        receive_time_ns: int,
    ) -> FreeFlightDetection | None:
        self._observe(
            "pose", position_m, time_ns=time_ns, receive_time_ns=receive_time_ns
        )
        return None

    def observe_twist(
        self,
        velocity_mps: Sequence[float],
        *,
        time_ns: int,
        receive_time_ns: int,
    ) -> FreeFlightDetection | None:
        added = self._observe(
            "twist", velocity_mps, time_ns=time_ns, receive_time_ns=receive_time_ns
        )
        if not added or self._detection is not None:
            return None
        self._detection = self._try_detect()
        return self._detection

    def _observe(
        self,
        source: MeasurementSource,
        measurement: Sequence[float],
        *,
        time_ns: int,
        receive_time_ns: int,
    ) -> bool:
        value = np.asarray(measurement, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{source} measurement must be a finite three-vector")
        for name, timestamp in (
            ("time_ns", time_ns),
            ("receive_time_ns", receive_time_ns),
        ):
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, int)
                or timestamp < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        latest = self._latest_source_time_ns[source]
        if latest is not None and time_ns <= latest:
            return False

        self._latest_source_time_ns[source] = time_ns
        self._measurements.append(
            BufferedMeasurement(
                source=source,
                time_ns=time_ns,
                receive_time_ns=receive_time_ns,
                value=(float(value[0]), float(value[1]), float(value[2])),
            )
        )
        newest_time_ns = max(
            timestamp
            for timestamp in self._latest_source_time_ns.values()
            if timestamp is not None
        )
        oldest_kept_ns = newest_time_ns - self.buffer_duration_ns
        self._measurements = [
            item for item in self._measurements if item.time_ns >= oldest_kept_ns
        ]
        return True

    @staticmethod
    def _rms(residuals: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))

    def _try_detect(self) -> FreeFlightDetection | None:
        latest_pose_ns = self._latest_source_time_ns["pose"]
        latest_twist_ns = self._latest_source_time_ns["twist"]
        if latest_pose_ns is None or latest_twist_ns is None:
            return None
        end_time_ns = min(latest_pose_ns, latest_twist_ns)
        start_time_ns = end_time_ns - self.fit_window_ns
        pose_items = [
            item
            for item in self._measurements
            if item.source == "pose" and start_time_ns <= item.time_ns <= end_time_ns
        ]
        twist_items = [
            item
            for item in self._measurements
            if item.source == "twist" and start_time_ns <= item.time_ns <= end_time_ns
        ]
        if (
            len(pose_items) < self.min_pose_samples
            or len(twist_items) < self.min_twist_samples
        ):
            self._reset_candidate()
            return None

        release_time_ns = min(pose_items[0].time_ns, twist_items[0].time_ns)
        covered_duration_ns = min(
            pose_items[-1].time_ns - pose_items[0].time_ns,
            twist_items[-1].time_ns - twist_items[0].time_ns,
        )
        if covered_duration_ns < round(0.8 * self.fit_window_ns):
            self._reset_candidate()
            return None

        gravity = np.array([0.0, 0.0, -self.gravity_mps2])
        twist_times_s = np.array(
            [(item.time_ns - release_time_ns) * 1e-9 for item in twist_items],
            dtype=float,
        )
        velocities = np.asarray([item.value for item in twist_items], dtype=float)
        if float(np.max(np.linalg.norm(velocities, axis=1))) < self.min_speed_mps:
            self._reset_candidate()
            return None
        initial_velocity_samples = velocities - twist_times_s[:, None] * gravity
        initial_velocity = np.mean(initial_velocity_samples, axis=0)
        velocity_residuals = velocities - (
            initial_velocity[None, :] + twist_times_s[:, None] * gravity
        )
        velocity_rms = self._rms(velocity_residuals)
        velocity_peak_error = float(
            np.max(np.linalg.norm(velocity_residuals, axis=1))
        )
        centered_twist_times_s = twist_times_s - float(np.mean(twist_times_s))
        time_energy = float(centered_twist_times_s @ centered_twist_times_s)
        if time_energy <= 0.0:
            self._reset_candidate()
            return None
        fitted_acceleration = (
            centered_twist_times_s[:, None] * velocities
        ).sum(axis=0) / time_energy
        acceleration_error = float(np.linalg.norm(fitted_acceleration - gravity))

        pose_times_s = np.array(
            [(item.time_ns - release_time_ns) * 1e-9 for item in pose_items],
            dtype=float,
        )
        positions = np.asarray([item.value for item in pose_items], dtype=float)
        initial_position_samples = positions - (
            pose_times_s[:, None] * initial_velocity[None, :]
            + 0.5 * pose_times_s[:, None] ** 2 * gravity[None, :]
        )
        initial_position = np.mean(initial_position_samples, axis=0)
        position_residuals = positions - (
            initial_position[None, :]
            + pose_times_s[:, None] * initial_velocity[None, :]
            + 0.5 * pose_times_s[:, None] ** 2 * gravity[None, :]
        )
        position_rms = self._rms(position_residuals)
        if (
            position_rms > self.max_position_rms_m
            or velocity_rms > self.max_velocity_rms_mps
            or velocity_peak_error > self.max_velocity_rms_mps
            or acceleration_error > self.max_acceleration_error_mps2
        ):
            self._reset_candidate()
            return None

        self._consecutive_passes += 1
        if self._candidate_release_time_ns is None:
            self._candidate_release_time_ns = release_time_ns
        else:
            self._candidate_release_time_ns = max(
                self._candidate_release_time_ns, release_time_ns
            )
        if self._consecutive_passes < self.confirmation_windows:
            return None

        confirmed_release_ns = self._candidate_release_time_ns
        replay = tuple(
            sorted(
                (
                    item
                    for item in self._measurements
                    if confirmed_release_ns <= item.time_ns <= end_time_ns
                ),
                key=lambda item: item.time_ns,
            )
        )
        return FreeFlightDetection(
            release_time_ns=confirmed_release_ns,
            confirmation_time_ns=end_time_ns,
            position_rms_m=position_rms,
            velocity_rms_mps=velocity_rms,
            acceleration_error_mps2=acceleration_error,
            measurements=replay,
        )

    def _reset_candidate(self) -> None:
        self._consecutive_passes = 0
        self._candidate_release_time_ns = None
