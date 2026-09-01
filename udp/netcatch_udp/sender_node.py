from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .config import UdpConfig, load_config
from .multicast import create_sender_socket, send_datagram
from .protocol import ProtocolError
from .sender_core import ErrorRecoveryGate, SenderCore


class UdpSenderNode(Node):
    def __init__(self, config: UdpConfig) -> None:
        super().__init__("netcatch_udp_sender")
        self._config = config
        self._core = SenderCore(config)
        self._log_gate = ErrorRecoveryGate(interval_s=1.0)
        self._sender = create_sender_socket(config)
        self._closed = False

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscription = self.create_subscription(
            String,
            config.prediction_topic,
            self._on_prediction,
            qos,
        )
        self._timer = self.create_timer(1.0 / config.publish_hz, self._on_tick)
        self.get_logger().info(
            "UDP sender ready: "
            f"topic={config.prediction_topic} group={config.multicast_group}:"
            f"{config.multicast_port} interface={config.multicast_interface_ip} "
            f"rate={config.publish_hz:g}Hz"
        )

    def _on_prediction(self, message: String) -> None:
        try:
            self._core.ingest_prediction(message.data)
        except ProtocolError as exc:
            if self._log_gate.should_log_error("prediction"):
                self.get_logger().warning(f"rejected prediction: {exc}")
        else:
            if self._log_gate.record_success("prediction"):
                self.get_logger().info("prediction input recovered")

    def _on_tick(self) -> None:
        try:
            packet = self._core.next_packet()
            if packet is None:
                return
            send_datagram(self._sender, self._config, packet)
        except (OSError, ProtocolError, ValueError) as exc:
            if self._log_gate.should_log_error("send"):
                self.get_logger().error(f"UDP send failed: {exc}")
        else:
            if self._log_gate.record_success("send"):
                self.get_logger().info("UDP send recovered")

    def destroy_node(self) -> bool:
        if not self._closed:
            self._sender.close()
            self._closed = True
        return super().destroy_node()


def build_argument_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).resolve().parents[1] / "config.json"
    parser = argparse.ArgumentParser(description="ROS-to-UDP NetCatch target sender")
    parser.add_argument("--config", default=str(default_config), help="path to UDP config JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    args, ros_args = build_argument_parser().parse_known_args(arguments)
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"sender configuration failed: {exc}", file=sys.stderr)
        return 2

    rclpy.init(args=ros_args)
    node = None
    try:
        node = UdpSenderNode(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        return 130
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
