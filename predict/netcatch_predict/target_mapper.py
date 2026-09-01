"""Map accepted interception points to safe payload targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class MappingResult:
    accepted: bool
    committed: bool
    reason: str
    target_m: np.ndarray

    @property
    def command_ready(self) -> bool:
        """True only when this sample may enter the prediction session."""

        return self.accepted and self.committed


def _finite_vector(name: str, value: Sequence[float], length: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {length}-vector")
    return result


class TargetMapper:
    def __init__(
        self,
        *,
        initial_payload_target_m: Sequence[float],
        payload_target_z_m: float,
        prediction_plane_z_m: float,
        consistency_samples: int,
        consistency_spread_max_m: float,
        deadband_m: float,
        jump_reject_m: float,
        workspace_xy_min_m: Sequence[float],
        workspace_xy_max_m: Sequence[float],
        usable_net_radius_m: float,
    ) -> None:
        initial = _finite_vector("initial_payload_target_m", initial_payload_target_m, 3)
        workspace_min = _finite_vector("workspace_xy_min_m", workspace_xy_min_m, 2)
        workspace_max = _finite_vector("workspace_xy_max_m", workspace_xy_max_m, 2)
        scalars = (
            payload_target_z_m,
            prediction_plane_z_m,
            consistency_spread_max_m,
            deadband_m,
            jump_reject_m,
            usable_net_radius_m,
        )
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("mapper parameters must be finite")
        if isinstance(consistency_samples, bool) or not isinstance(consistency_samples, int):
            raise ValueError("consistency_samples must be a positive integer")
        if consistency_samples <= 0:
            raise ValueError("consistency_samples must be a positive integer")
        if prediction_plane_z_m <= payload_target_z_m:
            raise ValueError("prediction plane must be above payload target")
        if any(value < 0.0 for value in (consistency_spread_max_m, deadband_m, usable_net_radius_m)):
            raise ValueError("mapper radii and thresholds must be nonnegative")
        if jump_reject_m <= 0.0 or deadband_m > jump_reject_m:
            raise ValueError("jump rejection must be positive and at least the deadband")
        if np.any(workspace_min >= workspace_max):
            raise ValueError("workspace minimum must be below maximum")
        if not math.isclose(float(initial[2]), payload_target_z_m, abs_tol=1e-12):
            raise ValueError("initial target z must equal payload target z")
        if np.any(initial[:2] < workspace_min) or np.any(initial[:2] > workspace_max):
            raise ValueError("initial target must lie inside workspace")

        self._initial_target = initial.copy()
        self._target_z = float(payload_target_z_m)
        self._plane_z = float(prediction_plane_z_m)
        self._consistency_samples = consistency_samples
        self._spread_max = float(consistency_spread_max_m)
        self._deadband = float(deadband_m)
        self._jump_reject = float(jump_reject_m)
        self._workspace_min = workspace_min
        self._workspace_max = workspace_max
        self.usable_net_radius_m = float(usable_net_radius_m)
        self.reset()

    @property
    def current_target_m(self) -> np.ndarray:
        return self._current_target.copy()

    @property
    def has_committed_target(self) -> bool:
        return self._committed

    @property
    def pending_count(self) -> int:
        return len(self._candidates)

    def reset(self) -> None:
        self._current_target = self._initial_target.copy()
        self._committed = False
        self._candidates: list[np.ndarray] = []

    def _result(self, accepted: bool, reason: str) -> MappingResult:
        return MappingResult(accepted, self._committed, reason, self._current_target.copy())

    def consider(self, intercept_m: Sequence[float]) -> MappingResult:
        try:
            intercept = _finite_vector("intercept_m", intercept_m, 3)
        except (TypeError, ValueError):
            return self._result(False, "malformed_intercept")
        if not math.isclose(float(intercept[2]), self._plane_z, abs_tol=1e-12):
            return self._result(False, "wrong_prediction_plane")
        xy = intercept[:2]
        if np.any(xy < self._workspace_min) or np.any(xy > self._workspace_max):
            return self._result(False, "outside_workspace")

        if not self._committed:
            self._candidates.append(xy.copy())
            if len(self._candidates) > self._consistency_samples:
                self._candidates.pop(0)
            if len(self._candidates) < self._consistency_samples:
                return self._result(True, "collecting_candidates")
            candidate_matrix = np.vstack(self._candidates)
            pairwise = candidate_matrix[:, None, :] - candidate_matrix[None, :, :]
            maximum_spread = float(np.linalg.norm(pairwise, axis=2).max())
            if maximum_spread > self._spread_max:
                return self._result(True, "inconsistent_candidates")
            self._current_target = np.array([xy[0], xy[1], self._target_z], dtype=float)
            self._committed = True
            return self._result(True, "initial_commit")

        distance = float(np.linalg.norm(xy - self._current_target[:2]))
        if distance > self._jump_reject:
            return self._result(False, "candidate_jump")
        if distance <= self._deadband:
            return self._result(True, "deadband_hold")
        self._current_target = np.array([xy[0], xy[1], self._target_z], dtype=float)
        return self._result(True, "target_updated")
