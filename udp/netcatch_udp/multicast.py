from __future__ import annotations

import math
import socket
from collections.abc import Callable
from typing import Any

from .config import UdpConfig


def validate_timeout_s(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("timeout_s must be a finite positive number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("timeout_s must be a finite positive number") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("timeout_s must be a finite positive number")
    return normalized


def create_sender_socket(
    config: UdpConfig,
    *,
    socket_factory: Callable[..., Any] = socket.socket,
) -> socket.socket:
    sender = socket_factory(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sender.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, config.multicast_ttl)
        sender.setsockopt(
            socket.IPPROTO_IP,
            socket.IP_MULTICAST_IF,
            socket.inet_aton(config.multicast_interface_ip),
        )
        sender.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
    except BaseException:
        sender.close()
        raise
    return sender


def send_datagram(sender: Any, config: UdpConfig, payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) > config.max_packet_bytes:
        raise ValueError(
            f"datagram too large: {len(payload)} > {config.max_packet_bytes} bytes"
        )
    sent = sender.sendto(payload, (config.multicast_group, config.multicast_port))
    if sent != len(payload):
        raise OSError(f"partial UDP datagram send: {sent} of {len(payload)} bytes")


def create_receiver_socket(
    config: UdpConfig,
    *,
    timeout_s: float,
    socket_factory: Callable[..., Any] = socket.socket,
) -> socket.socket:
    normalized_timeout_s = validate_timeout_s(timeout_s)
    receiver = socket_factory(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver.bind(("", config.multicast_port))
        membership = socket.inet_aton(config.multicast_group) + socket.inet_aton(
            config.multicast_interface_ip
        )
        receiver.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
        receiver.settimeout(normalized_timeout_s)
    except BaseException:
        receiver.close()
        raise
    return receiver
