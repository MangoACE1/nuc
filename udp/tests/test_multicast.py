from __future__ import annotations

import io
import json
import socket

import pytest

from netcatch_udp.config import UdpConfig


class FakeSocket:
    def __init__(self) -> None:
        self.options = {}
        self.closed = False
        self.sent = []
        self.bound_to = None
        self.timeout = None
        self.send_result = None

    def setsockopt(self, level, option, value) -> None:
        self.options[(level, option)] = value

    def getsockopt(self, level, option, buflen=None):
        del buflen
        return self.options[(level, option)]

    def close(self) -> None:
        self.closed = True

    def sendto(self, payload, destination):
        self.sent.append((payload, destination))
        return len(payload) if self.send_result is None else self.send_result

    def bind(self, address) -> None:
        self.bound_to = address

    def settimeout(self, timeout) -> None:
        self.timeout = timeout


def make_config(*, port: int = 15150, interface_ip: str = "127.0.0.1") -> UdpConfig:
    return UdpConfig(
        prediction_topic="/netcatch/dynamics/prediction",
        multicast_group="239.255.42.99",
        multicast_port=port,
        multicast_ttl=2,
        multicast_interface_ip=interface_ip,
        publish_hz=30.0,
        prediction_stale_s=0.15,
        max_packet_bytes=1200,
        frame_id="mocap_world_enu",
        object_name="obj1",
    )


def test_sender_socket_sets_ttl_interface_and_loopback():
    """Catches sending on the wrong NIC/scope or disabling same-host diagnostics."""
    try:
        from netcatch_udp.multicast import create_sender_socket
    except ModuleNotFoundError:
        pytest.fail("netcatch_udp.multicast sender socket is not implemented")

    sender = FakeSocket()
    constructor_args = []

    def socket_factory(*args):
        constructor_args.append(args)
        return sender

    created = create_sender_socket(make_config(), socket_factory=socket_factory)
    try:
        assert constructor_args == [(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)]
        assert created.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL) == 2
        assert created.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP) == 1
        assert created.getsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, 4) == socket.inet_aton(
            "127.0.0.1"
        )
    finally:
        created.close()


def test_send_datagram_calls_sendto_once_and_enforces_complete_bounded_write():
    """Catches duplicate sends, wrong destinations, oversize writes, or silent partial sends."""
    from netcatch_udp.multicast import send_datagram

    config = make_config()
    sender = FakeSocket()
    payload = b"wire-packet"

    send_datagram(sender, config, payload)

    assert sender.sent == [(payload, ("239.255.42.99", 15150))]

    with pytest.raises(ValueError, match="too large"):
        send_datagram(sender, config, b"x" * 1201)
    assert len(sender.sent) == 1

    sender.send_result = len(payload) - 1
    with pytest.raises(OSError, match="partial"):
        send_datagram(sender, config, payload)
    assert len(sender.sent) == 2


def test_receiver_socket_reuses_port_binds_and_joins_on_configured_interface():
    """Catches joining the wrong interface or omitting required receiver socket setup."""
    from netcatch_udp.multicast import create_receiver_socket

    receiver = FakeSocket()
    constructor_args = []

    def socket_factory(*args):
        constructor_args.append(args)
        return receiver

    created = create_receiver_socket(
        make_config(),
        timeout_s=0.25,
        socket_factory=socket_factory,
    )
    try:
        assert constructor_args == [(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)]
        assert created.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 1
        assert created.bound_to == ("", 15150)
        assert created.getsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP) == (
            socket.inet_aton("239.255.42.99") + socket.inet_aton("127.0.0.1")
        )
        assert created.timeout == 0.25
    finally:
        created.close()


def test_local_multicast_round_trip_when_environment_allows_socket_io():
    """Exercises a real loopback membership and one bounded multicast datagram."""
    from netcatch_udp.multicast import (
        create_receiver_socket,
        create_sender_socket,
        send_datagram,
    )

    try:
        reserve = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    except PermissionError as exc:
        pytest.skip(f"socket creation denied by test sandbox: {exc}")
    try:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    finally:
        reserve.close()

    config = make_config(port=port)
    receiver = sender = None
    try:
        receiver = create_receiver_socket(config, timeout_s=0.25)
        sender = create_sender_socket(config)
        send_datagram(sender, config, b"netcatch-multicast-smoke")
        payload, _source = receiver.recvfrom(1200)
    except (OSError, TimeoutError) as exc:
        pytest.skip(f"local multicast loopback unavailable: {exc}")
    finally:
        if sender is not None:
            sender.close()
        if receiver is not None:
            receiver.close()

    assert payload == b"netcatch-multicast-smoke"


class QueueReceiver:
    def __init__(self, items) -> None:
        self.items = list(items)

    def recvfrom(self, _max_bytes):
        if not self.items:
            raise socket.timeout("timed out")
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def make_wire_packet(*, seq: int = 0) -> bytes:
    from netcatch_udp.protocol import encode_wire_packet

    prediction = {
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
    return encode_wire_packet(
        prediction,
        source_boot_id="123e4567-e89b-42d3-a456-426614174000",
        seq=seq,
        send_time_ns=1_700_000_000_000_000_000 + seq,
        max_packet_bytes=1200,
        expected_frame_id="mocap_world_enu",
        expected_object_name="obj1",
    )


def test_receiver_loop_reports_malformed_then_prints_one_valid_text_packet():
    """Catches malformed input terminating the receiver or counting as a valid packet."""
    try:
        from netcatch_udp.receiver_cli import run_receiver
    except ModuleNotFoundError:
        pytest.fail("netcatch_udp.receiver_cli is not implemented")

    receiver = QueueReceiver(
        [
            (b"{}", ("10.0.0.9", 50000)),
            (make_wire_packet(), ("10.0.0.8", 50001)),
        ]
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    result = run_receiver(
        make_config(),
        receiver,
        count=1,
        json_output=False,
        stdout=stdout,
        stderr=stderr,
    )

    assert result == 0
    assert "malformed packet from 10.0.0.9:50000" in stderr.getvalue()
    line = stdout.getvalue().strip()
    assert "sender=10.0.0.8:50001" in line
    assert "boot=123e4567-e89b-42d3-a456-426614174000" in line
    assert "throw=7 seq=0 state=TRACKING valid=true" in line
    assert "target=[1.0,-0.1,0.4] ttc_s=0.2" in line


def test_receiver_loop_json_mode_includes_sender_and_validated_packet():
    """Catches JSON mode omitting sender identity or changing packet values."""
    from netcatch_udp.receiver_cli import run_receiver

    stdout = io.StringIO()
    result = run_receiver(
        make_config(),
        QueueReceiver([(make_wire_packet(seq=4), ("10.0.0.8", 50001))]),
        count=1,
        json_output=True,
        stdout=stdout,
        stderr=io.StringIO(),
    )

    rendered = json.loads(stdout.getvalue())
    assert result == 0
    assert rendered["sender"] == "10.0.0.8:50001"
    assert rendered["source_boot_id"] == "123e4567-e89b-42d3-a456-426614174000"
    assert rendered["seq"] == 4
    assert rendered["command_target_m"] == [1.0, -0.1, 0.4]


def test_receiver_loop_timeout_returns_nonzero_without_hanging():
    """Catches an empty receiver spinning forever after its configured socket timeout."""
    from netcatch_udp.receiver_cli import run_receiver

    stderr = io.StringIO()
    result = run_receiver(
        make_config(),
        QueueReceiver([]),
        count=1,
        json_output=False,
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert result == 1
    assert "receive timeout" in stderr.getvalue()


@pytest.mark.parametrize("literal", ["nan", "inf", "-inf", "0", "-0.1"])
def test_receiver_cli_rejects_nonfinite_or_nonpositive_timeout(literal):
    """Catches argparse forwarding a timeout that cannot bound a receive wait safely."""
    from netcatch_udp.receiver_cli import build_argument_parser

    with pytest.raises(SystemExit):
        build_argument_parser().parse_args(["--timeout-s", literal])


@pytest.mark.parametrize(
    "timeout_s",
    [float("nan"), float("inf"), float("-inf"), 0.0, -0.1, True, 10**400],
)
def test_receiver_socket_rejects_unsafe_timeout_before_socket_creation(timeout_s):
    """Catches programmatic callers forwarding unsafe timeout values to the OS socket."""
    from netcatch_udp.multicast import create_receiver_socket

    constructor_calls = []

    def socket_factory(*args):
        constructor_calls.append(args)
        return FakeSocket()

    with pytest.raises(ValueError, match="finite positive"):
        create_receiver_socket(
            make_config(),
            timeout_s=timeout_s,
            socket_factory=socket_factory,
        )
    assert constructor_calls == []
