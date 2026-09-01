"""Six-state ballistic propagation and descending-plane interception."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


CHI2_95_2D = 5.991


@dataclass(frozen=True, slots=True)
class BallisticPrediction:
    intercept_m: np.ndarray
    time_to_contact_s: float
    intercept_time_ns: int
    cov_xy_m2: np.ndarray
    xy_radius_95_m: float


def _dynamics(state: np.ndarray, gravity_mps2: float, drag_beta_m_inv: float) -> np.ndarray:
    velocity = state[3:]
    speed = float(np.linalg.norm(velocity))
    acceleration = np.array([0.0, 0.0, -gravity_mps2])
    if drag_beta_m_inv > 0.0 and speed > 0.0:
        acceleration -= drag_beta_m_inv * speed * velocity
    return np.concatenate((velocity, acceleration))


def _state_jacobian(state: np.ndarray, drag_beta_m_inv: float) -> np.ndarray:
    jacobian = np.zeros((6, 6), dtype=float)
    jacobian[:3, 3:] = np.eye(3)
    velocity = state[3:]
    speed = float(np.linalg.norm(velocity))
    if drag_beta_m_inv > 0.0 and speed > 1e-12:
        drag_jacobian = speed * np.eye(3) + np.outer(velocity, velocity) / speed
        jacobian[3:, 3:] = -drag_beta_m_inv * drag_jacobian
    return jacobian


def _rk4_state_transition_step(
    state: np.ndarray,
    transition: np.ndarray,
    step_s: float,
    gravity_mps2: float,
    drag_beta_m_inv: float,
) -> tuple[np.ndarray, np.ndarray]:
    def derivative(x: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return _dynamics(x, gravity_mps2, drag_beta_m_inv), _state_jacobian(
            x, drag_beta_m_inv
        ) @ phi

    k1_x, k1_phi = derivative(state, transition)
    k2_x, k2_phi = derivative(
        state + 0.5 * step_s * k1_x, transition + 0.5 * step_s * k1_phi
    )
    k3_x, k3_phi = derivative(
        state + 0.5 * step_s * k2_x, transition + 0.5 * step_s * k2_phi
    )
    k4_x, k4_phi = derivative(state + step_s * k3_x, transition + step_s * k3_phi)
    next_state = state + step_s * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0
    next_transition = transition + step_s * (
        k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi
    ) / 6.0
    return next_state, next_transition


def propagate_state_and_transition(
    state: np.ndarray,
    duration_s: float,
    *,
    gravity_mps2: float,
    drag_beta_m_inv: float,
    max_step_s: float = 0.005,
    initial_transition: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a state and its transition matrix over a bounded interval."""

    x = np.asarray(state, dtype=float).copy()
    if x.shape != (6,) or not np.all(np.isfinite(x)):
        raise ValueError("state must be a finite six-vector")
    if not math.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("duration_s must be finite and nonnegative")
    if not math.isfinite(gravity_mps2) or gravity_mps2 <= 0.0:
        raise ValueError("gravity_mps2 must be finite and positive")
    if not math.isfinite(drag_beta_m_inv) or drag_beta_m_inv < 0.0:
        raise ValueError("drag_beta_m_inv must be finite and nonnegative")
    if not math.isfinite(max_step_s) or max_step_s <= 0.0:
        raise ValueError("max_step_s must be finite and positive")
    transition = (
        np.eye(6, dtype=float)
        if initial_transition is None
        else np.asarray(initial_transition, dtype=float).copy()
    )
    if transition.shape != (6, 6) or not np.all(np.isfinite(transition)):
        raise ValueError("initial_transition must be a finite 6x6 matrix")
    if duration_s == 0.0:
        return x, transition

    if drag_beta_m_inv == 0.0:
        x[:3] += x[3:] * duration_s
        x[2] -= 0.5 * gravity_mps2 * duration_s * duration_s
        x[5] -= gravity_mps2 * duration_s
        local_transition = np.eye(6)
        local_transition[:3, 3:] = np.eye(3) * duration_s
        return x, local_transition @ transition

    remaining = duration_s
    while remaining > 0.0:
        step = min(max_step_s, remaining)
        x, transition = _rk4_state_transition_step(
            x, transition, step, gravity_mps2, drag_beta_m_inv
        )
        remaining -= step
        if remaining < 1e-15:
            remaining = 0.0
    return x, transition


def xy_radius_95(covariance_xy: np.ndarray) -> float:
    covariance = np.asarray(covariance_xy, dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise ValueError("XY covariance must be a finite 2x2 matrix")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] < -1e-10:
        raise ValueError("XY covariance must be positive semidefinite")
    return math.sqrt(CHI2_95_2D * max(0.0, float(eigenvalues[-1])))


def _prediction_from_crossing(
    state: np.ndarray,
    transition: np.ndarray,
    covariance: np.ndarray,
    time_to_contact_s: float,
    state_time_ns: int,
    plane_z_m: float,
) -> BallisticPrediction:
    intercept = state[:3].copy()
    intercept[2] = plane_z_m
    propagated_covariance = transition @ covariance @ transition.T
    propagated_covariance = 0.5 * (propagated_covariance + propagated_covariance.T)
    crossing_vz = float(state[5])
    if crossing_vz >= -1e-12:
        raise ValueError("event-surface covariance requires a descending crossing")
    event_projection = np.zeros((2, 6), dtype=float)
    event_projection[0, 0] = 1.0
    event_projection[1, 1] = 1.0
    event_projection[0, 2] = -float(state[3]) / crossing_vz
    event_projection[1, 2] = -float(state[4]) / crossing_vz
    xy_covariance = event_projection @ propagated_covariance @ event_projection.T
    xy_covariance = 0.5 * (xy_covariance + xy_covariance.T)
    compact_covariance = np.array(
        [xy_covariance[0, 0], xy_covariance[0, 1], xy_covariance[1, 1]], dtype=float
    )
    return BallisticPrediction(
        intercept_m=intercept,
        time_to_contact_s=float(time_to_contact_s),
        intercept_time_ns=state_time_ns + round(time_to_contact_s * 1e9),
        cov_xy_m2=compact_covariance,
        xy_radius_95_m=xy_radius_95(xy_covariance),
    )


def predict_descending_crossing(
    state: np.ndarray,
    covariance: np.ndarray,
    *,
    state_time_ns: int,
    plane_z_m: float,
    gravity_mps2: float,
    drag_beta_m_inv: float,
    max_step_s: float = 0.005,
    max_time_s: float = 5.0,
) -> BallisticPrediction | None:
    x0 = np.asarray(state, dtype=float)
    covariance0 = np.asarray(covariance, dtype=float)
    if x0.shape != (6,) or not np.all(np.isfinite(x0)):
        raise ValueError("state must be a finite six-vector")
    if covariance0.shape != (6, 6) or not np.all(np.isfinite(covariance0)):
        raise ValueError("covariance must be a finite 6x6 matrix")
    if not np.allclose(covariance0, covariance0.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(covariance0).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    if isinstance(state_time_ns, bool) or not isinstance(state_time_ns, int) or state_time_ns < 0:
        raise ValueError("state_time_ns must be a nonnegative integer")
    for name, value in (
        ("plane_z_m", plane_z_m),
        ("gravity_mps2", gravity_mps2),
        ("drag_beta_m_inv", drag_beta_m_inv),
        ("max_step_s", max_step_s),
        ("max_time_s", max_time_s),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if gravity_mps2 <= 0.0 or max_step_s <= 0.0 or max_time_s <= 0.0:
        raise ValueError("gravity, max step, and max time must be positive")
    if drag_beta_m_inv < 0.0:
        raise ValueError("drag beta must be nonnegative")

    if drag_beta_m_inv == 0.0:
        discriminant = x0[5] * x0[5] + 2.0 * gravity_mps2 * (x0[2] - plane_z_m)
        if discriminant < 0.0:
            return None
        time_to_contact = (x0[5] + math.sqrt(max(0.0, discriminant))) / gravity_mps2
        if time_to_contact <= 1e-12 or time_to_contact > max_time_s:
            return None
        crossing_state, transition = propagate_state_and_transition(
            x0,
            time_to_contact,
            gravity_mps2=gravity_mps2,
            drag_beta_m_inv=0.0,
            max_step_s=max_step_s,
        )
        if crossing_state[5] >= 0.0:
            return None
        return _prediction_from_crossing(
            crossing_state,
            transition,
            covariance0,
            time_to_contact,
            state_time_ns,
            plane_z_m,
        )

    elapsed = 0.0
    current_state = x0.copy()
    current_transition = np.eye(6)
    while elapsed < max_time_s:
        step = min(max_step_s, max_time_s - elapsed)
        previous_state = current_state
        previous_transition = current_transition
        current_state, current_transition = _rk4_state_transition_step(
            previous_state,
            previous_transition,
            step,
            gravity_mps2,
            drag_beta_m_inv,
        )
        if (
            previous_state[2] > plane_z_m
            and current_state[2] <= plane_z_m
            and current_state[5] < 0.0
        ):
            low_time = 0.0
            high_time = step
            low_z = float(previous_state[2])
            high_z = float(current_state[2])
            crossing_state = current_state
            crossing_transition = current_transition
            crossing_offset = high_time
            for _ in range(24):
                denominator = low_z - high_z
                if denominator <= 0.0:
                    trial_time = 0.5 * (low_time + high_time)
                else:
                    trial_time = low_time + (low_z - plane_z_m) * (
                        high_time - low_time
                    ) / denominator
                trial_time = min(high_time, max(low_time, trial_time))
                trial_state, trial_transition = _rk4_state_transition_step(
                    previous_state,
                    previous_transition,
                    trial_time,
                    gravity_mps2,
                    drag_beta_m_inv,
                )
                crossing_state = trial_state
                crossing_transition = trial_transition
                crossing_offset = trial_time
                if abs(float(trial_state[2]) - plane_z_m) <= 1e-12:
                    break
                if trial_state[2] > plane_z_m:
                    low_time, low_z = trial_time, float(trial_state[2])
                else:
                    high_time, high_z = trial_time, float(trial_state[2])
            time_to_contact = elapsed + crossing_offset
            return _prediction_from_crossing(
                crossing_state,
                crossing_transition,
                covariance0,
                time_to_contact,
                state_time_ns,
                plane_z_m,
            )
        elapsed += step
    return None
