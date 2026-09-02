from __future__ import annotations

import math

import numpy as np
import pytest


def test_zero_drag_crossing_matches_a_hand_derived_ballistic_solution() -> None:
    from netcatch_predict.ballistics import predict_descending_crossing

    state = np.array([0.1, -0.3, 2.0, 0.4, -0.2, 1.0])
    covariance = np.diag([0.01, 0.04, 0.001, 0.0, 0.0, 0.0])
    expected_t = (1.0 + math.sqrt(1.0 + 2.0 * 9.81 * (2.0 - 0.833))) / 9.81

    prediction = predict_descending_crossing(
        state,
        covariance,
        state_time_ns=2_000_000_000,
        plane_z_m=0.833,
        gravity_mps2=9.81,
        drag_beta_m_inv=0.0,
    )

    assert prediction is not None
    np.testing.assert_allclose(
        prediction.intercept_m,
        [0.1 + 0.4 * expected_t, -0.3 - 0.2 * expected_t, 0.833],
        atol=1e-12,
    )
    assert prediction.time_to_contact_s == pytest.approx(expected_t, abs=1e-12)
    assert prediction.intercept_time_ns == 2_000_000_000 + round(expected_t * 1e9)
    crossing_vz = -math.sqrt(1.0 + 2.0 * 9.81 * (2.0 - 0.833))
    event_x_from_z = -0.4 / crossing_vz
    event_y_from_z = 0.2 / crossing_vz
    expected_covariance = np.array(
        [
            [0.01 + event_x_from_z**2 * 0.001, event_x_from_z * event_y_from_z * 0.001],
            [event_x_from_z * event_y_from_z * 0.001, 0.04 + event_y_from_z**2 * 0.001],
        ]
    )
    np.testing.assert_allclose(
        prediction.cov_xy_m2,
        [expected_covariance[0, 0], expected_covariance[0, 1], expected_covariance[1, 1]],
        atol=1e-12,
    )
    assert prediction.xy_radius_95_m == pytest.approx(
        math.sqrt(5.991 * np.linalg.eigvalsh(expected_covariance)[-1]), abs=1e-12
    )


def test_crossing_is_future_and_descending_even_when_state_starts_below_plane_ascending() -> None:
    from netcatch_predict.ballistics import predict_descending_crossing

    state = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 5.0])
    prediction = predict_descending_crossing(
        state,
        np.eye(6) * 1e-4,
        state_time_ns=0,
        plane_z_m=0.833,
        gravity_mps2=9.81,
        drag_beta_m_inv=0.0,
    )

    expected_t = (5.0 + math.sqrt(25.0 - 2.0 * 9.81 * (0.833 - 0.5))) / 9.81
    assert prediction is not None
    assert prediction.time_to_contact_s == pytest.approx(expected_t, abs=1e-12)
    assert 5.0 - 9.81 * prediction.time_to_contact_s < 0.0


def test_descending_state_already_below_plane_has_no_future_crossing() -> None:
    from netcatch_predict.ballistics import predict_descending_crossing

    prediction = predict_descending_crossing(
        np.array([0.0, 0.0, 0.5, 0.1, 0.0, -1.0]),
        np.eye(6),
        state_time_ns=0,
        plane_z_m=0.833,
        gravity_mps2=9.81,
        drag_beta_m_inv=0.0,
    )

    assert prediction is None


def test_positive_drag_uses_stable_bounded_step_propagation_and_interpolation() -> None:
    from netcatch_predict.ballistics import predict_descending_crossing

    state = np.array([0.0, 0.0, 2.0, 2.0, 0.5, 1.0])
    covariance = np.diag([0.002, 0.003, 0.002, 0.01, 0.01, 0.01])
    no_drag = predict_descending_crossing(
        state,
        covariance,
        state_time_ns=1_000,
        plane_z_m=0.833,
        gravity_mps2=9.81,
        drag_beta_m_inv=0.0,
    )
    coarse = predict_descending_crossing(
        state,
        covariance,
        state_time_ns=1_000,
        plane_z_m=0.833,
        gravity_mps2=9.81,
        drag_beta_m_inv=0.2,
        max_step_s=0.02,
    )
    fine = predict_descending_crossing(
        state,
        covariance,
        state_time_ns=1_000,
        plane_z_m=0.833,
        gravity_mps2=9.81,
        drag_beta_m_inv=0.2,
        max_step_s=0.005,
    )

    assert no_drag is not None and coarse is not None and fine is not None
    assert fine.intercept_m[2] == 0.833
    assert fine.time_to_contact_s > 0.0
    assert fine.intercept_m[0] < no_drag.intercept_m[0]
    np.testing.assert_allclose(coarse.intercept_m, fine.intercept_m, atol=2e-5)
    assert coarse.time_to_contact_s == pytest.approx(fine.time_to_contact_s, abs=2e-5)
    assert np.all(np.isfinite(fine.cov_xy_m2))
    assert fine.xy_radius_95_m >= 0.0


@pytest.mark.parametrize("drag_beta_m_inv", [0.0, 0.2])
def test_event_surface_projection_turns_height_uncertainty_into_xy_uncertainty(
    drag_beta_m_inv: float,
) -> None:
    from netcatch_predict.ballistics import predict_descending_crossing

    covariance = np.zeros((6, 6))
    covariance[2, 2] = 0.04
    prediction = predict_descending_crossing(
        np.array([0.0, 0.0, 2.0, 2.0, 1.0, 0.0]),
        covariance,
        state_time_ns=0,
        plane_z_m=0.833,
        gravity_mps2=9.81,
        drag_beta_m_inv=drag_beta_m_inv,
    )

    assert prediction is not None
    pxx, pxy, pyy = prediction.cov_xy_m2
    assert pxx > 0.0
    assert pyy > 0.0
    assert pxy > 0.0
    projected = np.array([[pxx, pxy], [pxy, pyy]])
    assert np.linalg.eigvalsh(projected).min() >= -1e-12
    assert pxx * pyy - pxy * pxy >= -1e-12
    assert prediction.xy_radius_95_m > 0.0


def test_xy_radius_uses_largest_eigenvalue_of_full_xy_covariance() -> None:
    from netcatch_predict.ballistics import xy_radius_95

    covariance = np.array([[0.04, 0.03], [0.03, 0.04]])

    assert xy_radius_95(covariance) == pytest.approx(math.sqrt(5.991 * 0.07), abs=1e-12)


@pytest.mark.parametrize(
    ("state", "covariance"),
    [
        (np.array([0.0, 0.0, np.nan, 0.0, 0.0, 0.0]), np.eye(6)),
        (np.zeros(5), np.eye(6)),
        (np.zeros(6), np.eye(5)),
    ],
)
def test_ballistics_rejects_nonfinite_or_malformed_state_inputs(
    state: np.ndarray, covariance: np.ndarray
) -> None:
    from netcatch_predict.ballistics import predict_descending_crossing

    with pytest.raises(ValueError):
        predict_descending_crossing(
            state,
            covariance,
            state_time_ns=0,
            plane_z_m=0.833,
            gravity_mps2=9.81,
            drag_beta_m_inv=0.0,
        )


def test_plane_crossing_fraction_interpolates_descending_straddle() -> None:
    from netcatch_predict.ballistics import descending_plane_crossing_fraction

    # falls from z=1.0 to z=0.6; plane 0.8 is halfway down that segment.
    previous = np.array([0.0, 0.0, 1.0])
    current = np.array([1.0, 1.0, 0.6])
    assert descending_plane_crossing_fraction(previous, current, 0.8) == pytest.approx(
        0.5, abs=1e-12
    )
    fraction = descending_plane_crossing_fraction(previous, current, 0.8)
    crossing_xy = previous[:2] + fraction * (current[:2] - previous[:2])
    assert crossing_xy == pytest.approx([0.5, 0.5], abs=1e-12)


def test_plane_crossing_fraction_rejects_non_descending_or_non_straddling() -> None:
    from netcatch_predict.ballistics import descending_plane_crossing_fraction

    plane = 0.833
    cases = [
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.9])),  # both above
        (np.array([0.0, 0.0, 0.7]), np.array([0.0, 0.0, 0.5])),  # both below
        (np.array([0.0, 0.0, 0.7]), np.array([0.0, 0.0, 0.9])),  # rising through plane
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.9])),  # descending but no straddle
    ]
    for previous, current in cases:
        assert descending_plane_crossing_fraction(previous, current, plane) is None


def test_plane_crossing_fraction_exact_boundary_is_zero() -> None:
    from netcatch_predict.ballistics import descending_plane_crossing_fraction

    previous = np.array([2.0, 3.0, 0.833])
    current = np.array([4.0, 5.0, 0.5])
    assert descending_plane_crossing_fraction(previous, current, 0.833) == pytest.approx(
        0.0, abs=1e-12
    )
