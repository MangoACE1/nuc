from __future__ import annotations

import json

import pytest


VALID_PREDICTION_LITERAL = (
    '{"schema":"netcatch.prediction.v1","throw_id":7,"state":"TRACKING",'
    '"valid":true,"frame_id":"mocap_world_enu","object_name":"obj1",'
    '"reason":"tracking","pose_time_ns":1000000000,"twist_time_ns":1000000001,'
    '"state_time_ns":1000000002,"intercept_time_ns":1200000002,'
    '"time_to_contact_s":0.2,"prediction_plane_z_m":0.833,'
    '"intercept_m":[1.1,-0.2,0.833],"command_target_m":[1.0,-0.1,0.4],'
    '"hold_target_m":[0.0,0.0,0.4],"cov_xy_m2":[0.01,0.002,0.02],'
    '"xy_radius_95_m":0.3,"confidence":0.85}'
)


def test_decode_prediction_accepts_complete_finite_v1_literal():
    """Catches a decoder that drops or alters a valid predictor message."""
    try:
        from netcatch_udp.protocol import decode_prediction_json
    except ModuleNotFoundError:
        pytest.fail("netcatch_udp.protocol prediction decoder is not implemented")

    prediction = decode_prediction_json(
        VALID_PREDICTION_LITERAL,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
    )

    assert prediction == json.loads(VALID_PREDICTION_LITERAL)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema", "netcatch.prediction.v2"),
        ("throw_id", True),
        ("throw_id", 7.0),
        ("throw_id", -1),
        ("state", "FLYING"),
        ("valid", 1),
        ("frame_id", "map"),
        ("object_name", "obj2"),
        ("reason", 4),
        ("pose_time_ns", False),
        ("twist_time_ns", 1.5),
        ("state_time_ns", -1),
        ("intercept_time_ns", True),
        ("time_to_contact_s", -0.001),
        ("time_to_contact_s", float("inf")),
        ("prediction_plane_z_m", 0.8341),
        ("intercept_m", [1.1, -0.2]),
        ("intercept_m", [1.1, -0.2, 0.832]),
        ("command_target_m", [1.0, -0.1, 0.401]),
        ("hold_target_m", [0.0, 0.0, 0.399]),
        ("hold_target_m", [0.0, True, 0.4]),
        ("cov_xy_m2", [0.01, 0.002, -0.02]),
        ("cov_xy_m2", [0.01, 1.0, 0.02]),
        ("xy_radius_95_m", -0.1),
        ("confidence", -0.01),
        ("confidence", 1.01),
    ],
)
def test_decode_prediction_rejects_invalid_fields(field, bad_value):
    """Catches accepting wrong schema/types/ranges/geometry relationships."""
    from netcatch_udp.protocol import decode_prediction_json

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    prediction[field] = bad_value

    with pytest.raises(ValueError):
        decode_prediction_json(
            json.dumps(prediction),
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
        )


def test_decode_prediction_rejects_valid_true_outside_tracking():
    """Catches forwarding a non-TRACKING state as an actionable target."""
    from netcatch_udp.protocol import decode_prediction_json

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    prediction["state"] = "LOST"

    with pytest.raises(ValueError):
        decode_prediction_json(
            json.dumps(prediction),
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_decode_prediction_requires_exact_fields(mutation):
    """Catches silently accepting incomplete or extended prediction objects."""
    from netcatch_udp.protocol import decode_prediction_json

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    if mutation == "missing":
        del prediction["confidence"]
    else:
        prediction["debug_only"] = 123

    with pytest.raises(ValueError):
        decode_prediction_json(
            json.dumps(prediction),
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
        )


INVALID_PREDICTION_LITERALS = [
    VALID_PREDICTION_LITERAL.replace('"confidence":0.85', '"confidence":NaN'),
    VALID_PREDICTION_LITERAL.replace(
        '"throw_id":7', '"throw_id":7,"throw_id":8'
    ),
]


@pytest.mark.parametrize("literal", INVALID_PREDICTION_LITERALS)
def test_decode_prediction_rejects_nonfinite_or_duplicate_key_literal(literal):
    """Catches non-standard finite violations and ambiguous duplicate fields."""
    from netcatch_udp.protocol import decode_prediction_json

    with pytest.raises(ValueError):
        decode_prediction_json(
            literal,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
        )


@pytest.mark.parametrize(
    "state",
    ["WAIT_RELEASE", "TRACKING", "LOST", "RECOVERING", "DONE", "ERROR"],
)
def test_decode_prediction_accepts_all_declared_states_when_not_valid(state):
    """Catches accidentally narrowing the declared state machine."""
    from netcatch_udp.protocol import decode_prediction_json

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    prediction["state"] = state
    prediction["valid"] = False

    decoded = decode_prediction_json(
        json.dumps(prediction),
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
    )

    assert decoded["state"] == state
    assert decoded["valid"] is False


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("reason", "  "),
        ("pose_time_ns", 1_000_000_003),
        ("twist_time_ns", 1_000_000_003),
        ("intercept_time_ns", 1_000_000_001),
        ("intercept_time_ns", 1_200_000_004),
    ],
)
def test_decode_tracking_prediction_rejects_inconsistent_time_relationships(
    field, bad_value
):
    """Catches empty reasons, future observations, and inconsistent valid TTC timing."""
    from netcatch_udp.protocol import decode_prediction_json

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    prediction[field] = bad_value

    with pytest.raises(ValueError):
        decode_prediction_json(
            json.dumps(prediction),
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
        )


@pytest.mark.parametrize("rounding_delta_ns", [-1, 0, 1])
def test_decode_tracking_prediction_allows_one_ns_ttc_rounding_error(rounding_delta_ns):
    """Catches rejecting the predictor's documented integer-nanosecond rounding tolerance."""
    from netcatch_udp.protocol import decode_prediction_json

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    prediction["intercept_time_ns"] += rounding_delta_ns

    decoded = decode_prediction_json(
        json.dumps(prediction),
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
    )

    assert decoded["intercept_time_ns"] == 1_200_000_002 + rounding_delta_ns


VALID_BOOT_ID = "123e4567-e89b-42d3-a456-426614174000"
VALID_WIRE_LITERAL = (
    VALID_PREDICTION_LITERAL[:-1]
    + ',"magic":"NETCATCH_DYNAMIC_TARGET","version":1,'
    + f'"source_boot_id":"{VALID_BOOT_ID}","seq":0,'
    + '"send_time_ns":1700000000000000000}'
)


def test_decode_wire_packet_accepts_representative_v1_literal():
    """Catches a decoder that cannot consume a complete v1 datagram."""
    try:
        from netcatch_udp.protocol import decode_wire_packet
    except ImportError:
        pytest.fail("wire packet decoder is not implemented")

    packet = decode_wire_packet(
        VALID_WIRE_LITERAL.encode("utf-8"),
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
        max_packet_bytes=1200,
    )

    assert packet["magic"] == "NETCATCH_DYNAMIC_TARGET"
    assert packet["version"] == 1
    assert packet["source_boot_id"] == VALID_BOOT_ID
    assert packet["seq"] == 0
    assert packet["send_time_ns"] == 1_700_000_000_000_000_000
    assert packet["command_target_m"] == [1.0, -0.1, 0.4]


def test_encode_wire_packet_is_canonical_compact_utf8_and_round_trips():
    """Catches nondeterministic/non-compact encoding or envelope field loss."""
    from netcatch_udp.protocol import decode_wire_packet, encode_wire_packet

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    packet = encode_wire_packet(
        prediction,
        source_boot_id=VALID_BOOT_ID,
        seq=9,
        send_time_ns=1_700_000_000_000_000_123,
        max_packet_bytes=1200,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
    )
    expected = dict(prediction)
    expected.update(
        {
            "magic": "NETCATCH_DYNAMIC_TARGET",
            "version": 1,
            "source_boot_id": VALID_BOOT_ID,
            "seq": 9,
            "send_time_ns": 1_700_000_000_000_000_123,
        }
    )

    assert packet == json.dumps(
        expected, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert decode_wire_packet(
        packet,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
        max_packet_bytes=1200,
    ) == expected


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("magic", "NETCATCH_TARGET"),
        ("version", True),
        ("version", 2),
        ("source_boot_id", "123e4567-e89b-12d3-a456-426614174000"),
        ("source_boot_id", "not-a-uuid"),
        ("seq", True),
        ("seq", -1),
        ("send_time_ns", False),
        ("send_time_ns", 0),
    ],
)
def test_decode_wire_packet_rejects_invalid_envelope(field, bad_value):
    """Catches accepting a non-v1 identity, UUID, sequence, or wall timestamp."""
    from netcatch_udp.protocol import decode_wire_packet

    packet = json.loads(VALID_WIRE_LITERAL)
    packet[field] = bad_value

    with pytest.raises(ValueError):
        decode_wire_packet(
            json.dumps(packet).encode("utf-8"),
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=1200,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_decode_wire_packet_requires_exact_envelope_and_prediction_fields(mutation):
    """Catches accepting ambiguous wire extensions or missing metadata."""
    from netcatch_udp.protocol import decode_wire_packet

    packet = json.loads(VALID_WIRE_LITERAL)
    if mutation == "missing":
        del packet["magic"]
    else:
        packet["debug"] = True

    with pytest.raises(ValueError):
        decode_wire_packet(
            json.dumps(packet).encode("utf-8"),
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=1200,
        )


def test_wire_encoder_and_decoder_enforce_byte_budget():
    """Catches oversized datagrams on both sides of the network boundary."""
    from netcatch_udp.protocol import decode_wire_packet, encode_wire_packet

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    canonical = encode_wire_packet(
        prediction,
        source_boot_id=VALID_BOOT_ID,
        seq=0,
        send_time_ns=1,
        max_packet_bytes=1200,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
    )

    with pytest.raises(ValueError, match="too large"):
        encode_wire_packet(
            prediction,
            source_boot_id=VALID_BOOT_ID,
            seq=0,
            send_time_ns=1,
            max_packet_bytes=len(canonical) - 1,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
        )
    with pytest.raises(ValueError, match="too large"):
        decode_wire_packet(
            canonical,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=len(canonical) - 1,
        )


def test_prediction_decoder_normalizes_huge_integer_float_overflow():
    """Catches a bounded JSON integer escaping the callback as raw OverflowError."""
    from netcatch_udp.protocol import ProtocolError, decode_prediction_json

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    prediction["confidence"] = 10**400
    payload = json.dumps(prediction, separators=(",", ":")).encode("utf-8")
    assert len(payload) <= 1075

    with pytest.raises(ProtocolError, match="finite"):
        decode_prediction_json(
            payload,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
        )


def test_wire_decoder_normalizes_huge_integer_float_overflow():
    """Catches a sub-1200-byte wire packet escaping receiver handling as OverflowError."""
    from netcatch_udp.protocol import ProtocolError, decode_wire_packet

    packet = json.loads(VALID_WIRE_LITERAL)
    packet["confidence"] = 10**400
    payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
    assert len(payload) <= 1075

    with pytest.raises(ProtocolError, match="finite"):
        decode_wire_packet(
            payload,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=1200,
        )


def test_prediction_decoder_normalizes_ttc_nanosecond_scaling_overflow():
    """Catches finite TTC scaling to infinity and escaping as raw OverflowError."""
    from netcatch_udp.protocol import ProtocolError, decode_prediction_json

    prediction = json.loads(VALID_PREDICTION_LITERAL)
    prediction["time_to_contact_s"] = 1e308
    payload = json.dumps(prediction, separators=(",", ":")).encode("utf-8")
    assert len(payload) <= 1200

    with pytest.raises(ProtocolError, match="time_to_contact_s"):
        decode_prediction_json(
            payload,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
        )


def test_wire_decoder_normalizes_ttc_nanosecond_scaling_overflow():
    """Catches a complete bounded wire packet leaking TTC rounding OverflowError."""
    from netcatch_udp.protocol import ProtocolError, decode_wire_packet

    packet = json.loads(VALID_WIRE_LITERAL)
    packet["time_to_contact_s"] = 1e308
    payload = json.dumps(packet, separators=(",", ":")).encode("utf-8")
    assert len(payload) <= 1200

    with pytest.raises(ProtocolError, match="time_to_contact_s"):
        decode_wire_packet(
            payload,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=1200,
        )
