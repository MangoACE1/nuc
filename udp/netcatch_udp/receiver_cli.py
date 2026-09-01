from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from .config import UdpConfig, load_config
from .multicast import create_receiver_socket, validate_timeout_s
from .protocol import ProtocolError, decode_wire_packet


def _sender_text(address: tuple[str, int]) -> str:
    return f"{address[0]}:{address[1]}"


def _render_text(packet: dict[str, Any], sender: str) -> str:
    target = json.dumps(packet["command_target_m"], separators=(",", ":"))
    valid = str(packet["valid"]).lower()
    return (
        f"sender={sender} boot={packet['source_boot_id']} "
        f"throw={packet['throw_id']} seq={packet['seq']} "
        f"state={packet['state']} valid={valid} reason={packet['reason']} target={target} "
        f"ttc_s={packet['time_to_contact_s']}"
    )


def run_receiver(
    config: UdpConfig,
    receiver: Any,
    *,
    count: int,
    json_output: bool,
    stdout: TextIO,
    stderr: TextIO,
    changes_only: bool = False,
    receive_clock_ns: Callable[[], int] = time.time_ns,
) -> int:
    valid_count = 0
    last_display_key: tuple[object, ...] | None = None
    while count == 0 or valid_count < count:
        try:
            payload, address = receiver.recvfrom(config.max_packet_bytes + 1)
            receive_time_ns = receive_clock_ns()
        except socket.timeout:
            print("receive timeout", file=stderr, flush=True)
            return 1

        sender = _sender_text(address)
        try:
            packet = decode_wire_packet(
                payload,
                expected_frame_id=config.frame_id,
                expected_object_name=config.object_name,
                max_packet_bytes=config.max_packet_bytes,
            )
        except ProtocolError as exc:
            print(f"malformed packet from {sender}: {exc}", file=stderr, flush=True)
            continue

        display_key = (
            packet["source_boot_id"],
            packet["throw_id"],
            packet["state"],
            packet["valid"],
            packet["reason"],
        )
        should_display = not changes_only or display_key != last_display_key
        if should_display and json_output:
            rendered = dict(packet)
            rendered["sender"] = sender
            rendered["receive_time_ns"] = receive_time_ns
            print(
                json.dumps(
                    rendered,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                file=stdout,
                flush=True,
            )
        elif should_display:
            print(_render_text(packet, sender), file=stdout, flush=True)
        if should_display:
            last_display_key = display_key
        valid_count += 1
    return 0


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_float(value: str) -> float:
    try:
        return validate_timeout_s(float(value))
    except (OverflowError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite positive number") from exc


def build_argument_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[1] / "config.json"
    parser = argparse.ArgumentParser(description="Receive NetCatch multicast target packets")
    parser.add_argument("--config", default=str(default_config), help="path to UDP config JSON")
    parser.add_argument("--count", type=_nonnegative_int, default=0, help="valid packets; 0=infinite")
    parser.add_argument("--timeout-s", type=_positive_float, default=2.0, help="receive timeout")
    parser.add_argument("--json", action="store_true", dest="json_output", help="print JSON lines")
    parser.add_argument(
        "--changes-only",
        action="store_true",
        help="print only when boot, throw, state, valid, or reason changes",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        receiver = create_receiver_socket(config, timeout_s=args.timeout_s)
    except (OSError, ValueError) as exc:
        print(f"receiver setup failed: {exc}", file=sys.stderr)
        return 2

    try:
        return run_receiver(
            config,
            receiver,
            count=args.count,
            json_output=args.json_output,
            changes_only=args.changes_only,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except KeyboardInterrupt:
        print("receiver stopped", file=sys.stderr)
        return 130
    finally:
        receiver.close()


if __name__ == "__main__":
    raise SystemExit(main())
