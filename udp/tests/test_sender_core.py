from __future__ import annotations

import json
from uuid import UUID

import pytest

from netcatch_udp.config import UdpConfig


BOOT_ID = "123e4567-e89b-42d3-a456-426614174000"


def make_config() -> UdpConfig:
    return UdpConfig(
        prediction_topic="/netcatch/dynamics/prediction",
        multicast_group="239.255.42.99",
        multicast_port=15150,
        multicast_ttl=1,
        multicast_interface_ip="0.0.0.0",
        publish_hz=30.0,
        prediction_stale_s=0.15,
        max_packet_bytes=1200,
        frame_id="mocap_world_enu",
        object_name="obj1",
    )


def make_prediction(**updates):
    value = {
        "schema": "netcatch.prediction.v1",
        "throw_id": 7,
        "state": "TRACKING",
        "valid": True,
        "frame_id": "mocap_world_enu",
        "object_name": "obj1",
        "reason": "tracking",
        "pose_time_ns": 1_000_000_000,
        "twist_time_ns": 1_000_000_001,
        "state_time_ns": 1_000_000_002,
        "intercept_time_ns": 1_200_000_002,
        "time_to_contact_s": 0.2,
        "prediction_plane_z_m": 0.833,
        "intercept_m": [1.1, -0.2, 0.833],
        "command_target_m": [1.0, -0.1, 0.4],
        "hold_target_m": [0.0, 0.0, 0.4],
        "cov_xy_m2": [0.01, 0.002, 0.02],
        "xy_radius_95_m": 0.3,
        "confidence": 0.85,
    }
    value.update(updates)
    return value


def test_next_packet_is_none_until_a_prediction_arrives():
    """Catches emitting an invented heartbeat before any predictor input."""
    try:
        from netcatch_udp.sender_core import SenderCore
    except ModuleNotFoundError:
        pytest.fail("netcatch_udp.sender_core is not implemented")

    core = SenderCore(make_config(), source_boot_id=BOOT_ID)

    assert core.next_packet(
        now_monotonic_ns=1_000_000_000,
        send_time_ns=1_700_000_000_000_000_000,
    ) is None


def test_fresh_prediction_is_repeated_with_stable_boot_id_and_increasing_seq():
    """Catches one-shot sending, changing process identity, or reused sequence IDs."""
    from netcatch_udp.protocol import decode_wire_packet
    from netcatch_udp.sender_core import SenderCore

    core = SenderCore(make_config(), source_boot_id=BOOT_ID)
    core.ingest_prediction(
        json.dumps(make_prediction()),
        received_monotonic_ns=10_000_000_000,
    )

    first = core.next_packet(
        now_monotonic_ns=10_010_000_000,
        send_time_ns=1_700_000_000_000_000_001,
    )
    second = core.next_packet(
        now_monotonic_ns=10_020_000_000,
        send_time_ns=1_700_000_000_000_000_002,
    )
    decoded = [
        decode_wire_packet(
            packet,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=1200,
        )
        for packet in (first, second)
    ]

    assert [packet["seq"] for packet in decoded] == [0, 1]
    assert [packet["source_boot_id"] for packet in decoded] == [BOOT_ID, BOOT_ID]
    assert [packet["state"] for packet in decoded] == ["TRACKING", "TRACKING"]
    assert [packet["valid"] for packet in decoded] == [True, True]
    assert decoded[0]["command_target_m"] == [1.0, -0.1, 0.4]


def test_invalid_prediction_does_not_replace_cached_prediction():
    """Catches poisoning the last known-good cache with a malformed callback."""
    from netcatch_udp.protocol import decode_wire_packet
    from netcatch_udp.sender_core import SenderCore

    core = SenderCore(make_config(), source_boot_id=BOOT_ID)
    core.ingest_prediction(
        json.dumps(make_prediction()),
        received_monotonic_ns=10_000_000_000,
    )
    with pytest.raises(ValueError):
        core.ingest_prediction(
            json.dumps(make_prediction(frame_id="map", throw_id=99)),
            received_monotonic_ns=10_001_000_000,
        )

    packet = core.next_packet(
        now_monotonic_ns=10_010_000_000,
        send_time_ns=1_700_000_000_000_000_001,
    )
    decoded = decode_wire_packet(
        packet,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
        max_packet_bytes=1200,
    )

    assert decoded["throw_id"] == 7


def test_default_process_boot_id_is_generated_once_as_uuid4():
    """Catches a missing, non-v4, or per-datagram generated sender boot ID."""
    from netcatch_udp.protocol import decode_wire_packet
    from netcatch_udp.sender_core import SenderCore

    core = SenderCore(make_config())
    core.ingest_prediction(
        json.dumps(make_prediction()),
        received_monotonic_ns=10_000_000_000,
    )
    packets = [
        core.next_packet(
            now_monotonic_ns=10_001_000_000 + index,
            send_time_ns=1_700_000_000_000_000_001 + index,
        )
        for index in range(2)
    ]
    boot_ids = [
        decode_wire_packet(
            packet,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=1200,
        )["source_boot_id"]
        for packet in packets
    ]

    assert boot_ids[0] == boot_ids[1]
    assert UUID(boot_ids[0]).version == 4


def test_stale_boundary_uses_monotonic_age_and_preserves_diagnostic_targets():
    """Catches early/late stale switching or reusing an actionable stale target."""
    from netcatch_udp.protocol import decode_wire_packet
    from netcatch_udp.sender_core import SenderCore

    receipt_ns = 10_000_000_000
    core = SenderCore(make_config(), source_boot_id=BOOT_ID)
    core.ingest_prediction(
        json.dumps(make_prediction()),
        received_monotonic_ns=receipt_ns,
    )

    boundary_packet = core.next_packet(
        now_monotonic_ns=receipt_ns + 150_000_000,
        send_time_ns=1_700_000_000_000_000_001,
    )
    stale_packet = core.next_packet(
        now_monotonic_ns=receipt_ns + 150_000_001,
        send_time_ns=1_700_000_000_000_000_002,
    )
    boundary = decode_wire_packet(
        boundary_packet,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
        max_packet_bytes=1200,
    )
    stale = decode_wire_packet(
        stale_packet,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
        max_packet_bytes=1200,
    )

    assert (boundary["state"], boundary["valid"]) == ("TRACKING", True)
    assert (stale["state"], stale["valid"], stale["reason"]) == (
        "ERROR",
        False,
        "prediction_stale",
    )
    assert stale["time_to_contact_s"] == 0
    assert stale["command_target_m"] == [1.0, -0.1, 0.4]
    assert stale["hold_target_m"] == [0.0, 0.0, 0.4]
    assert [boundary["seq"], stale["seq"]] == [0, 1]


def test_wall_clock_jump_does_not_make_a_fresh_prediction_stale():
    """Catches using send_time_ns instead of monotonic time for cache age."""
    from netcatch_udp.protocol import decode_wire_packet
    from netcatch_udp.sender_core import SenderCore

    receipt_ns = 10_000_000_000
    core = SenderCore(make_config(), source_boot_id=BOOT_ID)
    core.ingest_prediction(
        json.dumps(make_prediction()),
        received_monotonic_ns=receipt_ns,
    )
    packets = [
        core.next_packet(
            now_monotonic_ns=receipt_ns + 149_000_000,
            send_time_ns=8_000_000_000_000_000_000,
        ),
        core.next_packet(
            now_monotonic_ns=receipt_ns + 149_000_001,
            send_time_ns=1,
        ),
    ]

    states = [
        decode_wire_packet(
            packet,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=1200,
        )["state"]
        for packet in packets
    ]

    assert states == ["TRACKING", "TRACKING"]


def test_fresh_callback_replaces_stale_error_on_next_tick():
    """Catches latching stale ERROR after a newly validated callback arrives."""
    from netcatch_udp.protocol import decode_wire_packet
    from netcatch_udp.sender_core import SenderCore

    core = SenderCore(make_config(), source_boot_id=BOOT_ID)
    core.ingest_prediction(
        json.dumps(make_prediction()),
        received_monotonic_ns=10_000_000_000,
    )
    stale_packet = core.next_packet(
        now_monotonic_ns=10_151_000_000,
        send_time_ns=1_700_000_000_000_000_001,
    )
    core.ingest_prediction(
        json.dumps(
            make_prediction(
                throw_id=8,
                command_target_m=[1.5, 0.25, 0.4],
                hold_target_m=[0.2, 0.1, 0.4],
            )
        ),
        received_monotonic_ns=10_152_000_000,
    )
    fresh_packet = core.next_packet(
        now_monotonic_ns=10_153_000_000,
        send_time_ns=1_700_000_000_000_000_002,
    )
    stale, fresh = [
        decode_wire_packet(
            packet,
            expected_frame_id="mocap_world_enu",
            expected_object_name="obj1",
            max_packet_bytes=1200,
        )
        for packet in (stale_packet, fresh_packet)
    ]

    assert (stale["state"], stale["valid"]) == ("ERROR", False)
    assert (fresh["state"], fresh["valid"], fresh["throw_id"]) == (
        "TRACKING",
        True,
        8,
    )
    assert fresh["command_target_m"] == [1.5, 0.25, 0.4]


def test_runtime_clock_callables_timestamp_receipt_and_tick():
    """Catches an adapter path that cannot use real clock callables without test overrides."""
    from netcatch_udp.protocol import decode_wire_packet
    from netcatch_udp.sender_core import SenderCore

    monotonic_values = iter([10_000_000_000, 10_010_000_000])
    core = SenderCore(
        make_config(),
        source_boot_id=BOOT_ID,
        monotonic_ns=lambda: next(monotonic_values),
        time_ns=lambda: 1_700_000_000_000_000_123,
    )
    core.ingest_prediction(json.dumps(make_prediction()))

    packet = core.next_packet()
    decoded = decode_wire_packet(
        packet,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
        max_packet_bytes=1200,
    )

    assert decoded["state"] == "TRACKING"
    assert decoded["send_time_ns"] == 1_700_000_000_000_000_123


def test_error_recovery_gate_throttles_repeated_class_to_one_hz():
    """Catches repeated prediction errors flooding logs faster than the gate interval."""
    from netcatch_udp.sender_core import ErrorRecoveryGate

    times = iter([0, 999_999_999, 1_000_000_000])
    gate = ErrorRecoveryGate(interval_s=1.0, monotonic_ns=lambda: next(times))

    assert gate.should_log_error("prediction") is True
    assert gate.should_log_error("prediction") is False
    assert gate.should_log_error("prediction") is True


def test_error_recovery_gate_keeps_prediction_and_send_classes_independent():
    """Catches one noisy error class suppressing the first report from another class."""
    from netcatch_udp.sender_core import ErrorRecoveryGate

    gate = ErrorRecoveryGate(interval_s=1.0)

    assert gate.should_log_error("prediction", now_monotonic_ns=100) is True
    assert gate.should_log_error("send", now_monotonic_ns=101) is True
    assert gate.should_log_error("prediction", now_monotonic_ns=102) is False
    assert gate.should_log_error("send", now_monotonic_ns=103) is False


def test_error_recovery_gate_reports_only_first_success_and_resets_incident():
    """Catches repeated recovery spam or suppressing a new incident after recovery."""
    from netcatch_udp.sender_core import ErrorRecoveryGate

    gate = ErrorRecoveryGate(interval_s=1.0)
    assert gate.should_log_error("send", now_monotonic_ns=100) is True

    assert gate.record_success("send") is True
    assert gate.record_success("send") is False
    assert gate.should_log_error("send", now_monotonic_ns=200) is True
    assert gate.record_success("send") is True
