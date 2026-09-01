from __future__ import annotations

import json
import pytest


def _valid_packet() -> dict[str, object]:
    return {
        "schema": "netcatch.prediction.v1",
        "throw_id": 7,
        "state": "TRACKING",
        "valid": True,
        "frame_id": "mocap_world_enu",
        "object_name": "obj1",
        "reason": "prediction_accepted",
        "pose_time_ns": 1_000_000_000,
        "twist_time_ns": 1_001_000_000,
        "state_time_ns": 1_001_000_000,
        "intercept_time_ns": 1_451_000_000,
        "time_to_contact_s": 0.45,
        "prediction_plane_z_m": 0.833,
        "intercept_m": [0.2, -0.1, 0.833],
        "command_target_m": [0.2, -0.1, 0.4],
        "hold_target_m": [-0.6, 0.6, 0.4],
        "cov_xy_m2": [0.01, 0.002, 0.02],
        "xy_radius_95_m": 0.35,
        "confidence": 0.5,
    }


def test_schema_round_trip_is_strict_and_preserves_every_v1_field() -> None:
    from netcatch_predict.schema import (
        PREDICTION_V1_KEYS,
        decode_prediction_packet,
        encode_prediction_packet,
    )

    encoded = encode_prediction_packet(_valid_packet())
    decoded = decode_prediction_packet(encoded)

    assert set(decoded) == PREDICTION_V1_KEYS
    assert decoded == _valid_packet()
    assert json.loads(encoded) == _valid_packet()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_schema_rejects_nonfinite_values_at_any_depth(bad_value: float) -> None:
    from netcatch_predict.schema import SchemaError, encode_prediction_packet

    packet = _valid_packet()
    packet["intercept_m"] = [0.2, bad_value, 0.833]

    with pytest.raises(SchemaError, match="finite"):
        encode_prediction_packet(packet)


def test_schema_decoder_rejects_nonstandard_json_nan() -> None:
    from netcatch_predict.schema import SchemaError, decode_prediction_packet

    raw = json.dumps(_valid_packet()).replace("0.35", "NaN")

    with pytest.raises(SchemaError, match="NaN|finite|constant"):
        decode_prediction_packet(raw)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_schema_rejects_missing_and_extra_fields(mutation: str) -> None:
    from netcatch_predict.schema import SchemaError, validate_prediction_packet

    packet = _valid_packet()
    if mutation == "missing":
        packet.pop("reason")
    else:
        packet["debug"] = "not part of v1"

    with pytest.raises(SchemaError, match="fields"):
        validate_prediction_packet(packet)


@pytest.mark.parametrize("state", ["WAIT_RELEASE", "LOST", "DONE"])
def test_wait_lost_and_done_packets_must_be_finite_and_invalid(state: str) -> None:
    from netcatch_predict.schema import SchemaError, validate_prediction_packet

    packet = _valid_packet()
    packet.update(
        {
            "state": state,
            "valid": False,
            "reason": state.lower(),
            "pose_time_ns": 0,
            "twist_time_ns": 0,
            "state_time_ns": 0,
            "intercept_time_ns": 0,
            "time_to_contact_s": 0.0,
            "intercept_m": [0.0, 0.0, 0.833],
            "cov_xy_m2": [0.0, 0.0, 0.0],
            "xy_radius_95_m": 0.0,
            "confidence": 0.0,
        }
    )
    assert validate_prediction_packet(packet)["valid"] is False

    packet["valid"] = True
    with pytest.raises(SchemaError, match="valid=false"):
        validate_prediction_packet(packet)


def test_schema_enforces_fixed_plane_and_payload_target_heights() -> None:
    from netcatch_predict.schema import SchemaError, validate_prediction_packet

    packet = _valid_packet()
    packet["intercept_m"] = [0.2, -0.1, 0.832]
    with pytest.raises(SchemaError, match="intercept.*plane"):
        validate_prediction_packet(packet)

    packet = _valid_packet()
    packet["command_target_m"] = [0.2, -0.1, 0.41]
    with pytest.raises(SchemaError, match="target.*0.4"):
        validate_prediction_packet(packet)


def test_confidence_is_diagnostic_and_zero_without_a_measured_usable_radius() -> None:
    from netcatch_predict.schema import confidence_from_radius

    assert confidence_from_radius(0.01, 0.0) == 0.0
    assert confidence_from_radius(0.25, 0.5) == 0.5
    assert confidence_from_radius(0.75, 0.5) == 0.0
    assert confidence_from_radius(0.0, 0.5) == 1.0


def test_schema_rejects_non_psd_xy_covariance() -> None:
    from netcatch_predict.schema import SchemaError, validate_prediction_packet

    packet = _valid_packet()
    packet["cov_xy_m2"] = [0.01, 0.02, 0.01]

    with pytest.raises(SchemaError, match="covariance"):
        validate_prediction_packet(packet)
