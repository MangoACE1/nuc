"""ROS 2 adapter for the NumPy-only NetCatch prediction core."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .ballistics import (
    BallisticPrediction,
    descending_plane_crossing_fraction,
    predict_descending_crossing,
)
from .csv_logger import CsvDebugLogger
from .config import PredictConfig, load_config
from .estimator import EstimatorSnapshot, StateEstimator
from .measurement_time import resolve_measurement_time_ns
from .release_detector import (
    BufferedMeasurement,
    FreeFlightDetection,
    ReleaseDetector,
)
from .schema import confidence_from_radius, encode_prediction_packet
from .session import PredictionSession, SessionState
from .source_health import SourceHealth
from .target_mapper import TargetMapper


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
ARM_SERVICE = "/netcatch/dynamics/arm"
RELEASE_SERVICE = "/netcatch/dynamics/release"
REARM_SERVICE = "/netcatch/dynamics/rearm"
CANCEL_SERVICE = "/netcatch/dynamics/cancel"


def _measurement_time_ns(
    message: PoseStamped | TwistStamped, *, receive_time_ns: int
) -> int:
    return resolve_measurement_time_ns(
        stamp_sec=int(message.header.stamp.sec),
        stamp_nanosec=int(message.header.stamp.nanosec),
        receive_time_ns=receive_time_ns,
    )


class PredictionNode(Node):
    """Thin single-threaded transport adapter around the pure core."""

    def __init__(self, config: PredictConfig, *, csv_log_path: str | None = None) -> None:
        super().__init__("netcatch_dynamics_predict")
        self.config = config
        self.estimator = StateEstimator(
            gravity_mps2=config.gravity_mps2,
            drag_beta_m_inv=config.drag_beta_m_inv,
            pose_position_std_m=config.pose_position_std_m,
            twist_velocity_std_mps=config.twist_velocity_std_mps,
            process_acceleration_std_mps2=config.process_acceleration_std_mps2,
            pose_innovation_gate_sigma=config.pose_innovation_gate_sigma,
            twist_innovation_gate_sigma=config.twist_innovation_gate_sigma,
            rk4_max_step_s=config.rk4_max_step_s,
            estimator_replay_history_samples=config.estimator_replay_history_samples,
        )
        self.mapper = TargetMapper(
            initial_payload_target_m=config.initial_payload_target_m,
            payload_target_z_m=config.payload_target_z_m,
            prediction_plane_z_m=config.prediction_plane_z_m,
            consistency_samples=config.consistency_samples,
            consistency_spread_max_m=config.consistency_spread_max_m,
            deadband_m=config.deadband_m,
            jump_reject_m=config.jump_reject_m,
            workspace_xy_min_m=config.workspace_xy_min_m,
            workspace_xy_max_m=config.workspace_xy_max_m,
            usable_net_radius_m=config.usable_net_radius_m,
        )
        self.session = PredictionSession(
            active_tracking_budget_s=config.active_tracking_budget_s,
            recovery_good_samples=config.recovery_good_samples,
            intercept_deadline_grace_s=config.intercept_deadline_grace_s,
            initial_hold_target_m=config.initial_payload_target_m,
            payload_target_z_m=config.payload_target_z_m,
        )
        self.release_detector = ReleaseDetector(
            gravity_mps2=config.gravity_mps2,
            buffer_duration_s=config.release_buffer_s,
            fit_window_s=config.release_fit_window_s,
            min_pose_samples=config.release_min_pose_samples,
            min_twist_samples=config.release_min_twist_samples,
            min_speed_mps=config.release_min_speed_mps,
            max_position_rms_m=config.release_max_position_rms_m,
            max_velocity_rms_mps=config.release_max_velocity_rms_mps,
            max_acceleration_error_mps2=config.release_max_acceleration_error_mps2,
            confirmation_windows=config.release_confirmation_windows,
        )

        self._payload_position_m = np.asarray(config.initial_payload_target_m, dtype=float)
        self._last_processed_counts = (0, 0)
        self._latest_estimate: EstimatorSnapshot | None = None
        self._latest_prediction: BallisticPrediction | None = None
        self._latest_prediction_valid = False
        self._packet_reason = "waiting_for_arm"
        self._pose_health = SourceHealth("pose")
        self._twist_health = SourceHealth("twist")
        self._object_subscriptions_active = False
        self._pose_subscription = None
        self._twist_subscription = None

        self._last_raw_pose: np.ndarray | None = None
        self._prev_raw_pose_time_ns: int | None = None
        self._actual_capture_m: np.ndarray | None = None
        self._actual_capture_time_ns: int | None = None
        self._ekf_epoch_start_monotonic_s: float | None = None
        self.csv_logger: CsvDebugLogger | None = None
        if csv_log_path:
            try:
                self.csv_logger = CsvDebugLogger(csv_log_path)
            except OSError as exc:
                self.get_logger().warning(
                    f"csv logging disabled: cannot open {csv_log_path}: {exc}"
                )

        self._publisher = self.create_publisher(String, config.output_topic, 10)
        self._payload_subscription = self.create_subscription(
            PoseStamped, config.payload_pose_topic, self._on_payload_pose, 10
        )
        self._arm_service = self.create_service(Trigger, ARM_SERVICE, self._on_arm)
        self._release_service = self.create_service(Trigger, RELEASE_SERVICE, self._on_release)
        self._rearm_service = self.create_service(Trigger, REARM_SERVICE, self._on_rearm)
        self._cancel_service = self.create_service(Trigger, CANCEL_SERVICE, self._on_cancel)
        self._create_object_subscriptions()
        self._publish_timer = self.create_timer(1.0 / config.publish_rate_hz, self._on_publish_timer)
        self.get_logger().info(
            f"waiting to arm automatic free-flight detection; "
            f"object={config.object_name}, frame={config.frame_id}"
        )

    def _create_object_subscriptions(self) -> None:
        if self._object_subscriptions_active:
            return
        self._pose_subscription = self.create_subscription(
            PoseStamped, self.config.pose_topic, self._on_object_pose, 10
        )
        self._twist_subscription = self.create_subscription(
            TwistStamped, self.config.twist_topic, self._on_object_twist, 10
        )
        self._object_subscriptions_active = True

    def _destroy_object_subscriptions(self) -> None:
        if not self._object_subscriptions_active:
            return
        if self._pose_subscription is not None:
            self.destroy_subscription(self._pose_subscription)
            self._pose_subscription = None
        if self._twist_subscription is not None:
            self.destroy_subscription(self._twist_subscription)
            self._twist_subscription = None
        self._object_subscriptions_active = False

    def _on_object_pose(self, message: PoseStamped) -> None:
        if not self.session.can_accept_object_update:
            return
        position = message.pose.position
        receive_time_ns = int(self.get_clock().now().nanoseconds)
        time_ns = _measurement_time_ns(message, receive_time_ns=receive_time_ns)
        value = [position.x, position.y, position.z]
        previous_pose = self._last_raw_pose
        previous_time_ns = self._prev_raw_pose_time_ns
        self._last_raw_pose = np.asarray(value, dtype=float)
        self._prev_raw_pose_time_ns = time_ns
        self._update_actual_capture(previous_pose, previous_time_ns, time_ns)
        if self.session.state is SessionState.WAIT_RELEASE:
            if self.session.is_armed:
                self.release_detector.observe_pose(
                    value,
                    time_ns=time_ns,
                    receive_time_ns=receive_time_ns,
                )
            return
        self._apply_estimator_measurement(
            BufferedMeasurement(
                source="pose",
                time_ns=time_ns,
                receive_time_ns=receive_time_ns,
                value=(float(position.x), float(position.y), float(position.z)),
            )
        )

    def _on_object_twist(self, message: TwistStamped) -> None:
        if not self.session.can_accept_object_update:
            return
        velocity = message.twist.linear
        receive_time_ns = int(self.get_clock().now().nanoseconds)
        time_ns = _measurement_time_ns(message, receive_time_ns=receive_time_ns)
        value = [velocity.x, velocity.y, velocity.z]
        if self.session.state is SessionState.WAIT_RELEASE:
            if self.session.is_armed:
                detection = self.release_detector.observe_twist(
                    value,
                    time_ns=time_ns,
                    receive_time_ns=receive_time_ns,
                )
                if detection is not None:
                    self._activate_automatic_release(detection)
            return
        self._apply_estimator_measurement(
            BufferedMeasurement(
                source="twist",
                time_ns=time_ns,
                receive_time_ns=receive_time_ns,
                value=(float(velocity.x), float(velocity.y), float(velocity.z)),
            )
        )

    def _update_actual_capture(
        self,
        previous_pose: np.ndarray | None,
        previous_time_ns: int | None,
        current_time_ns: int,
    ) -> None:
        """Latch where the object actually descends through the prediction plane.

        Interpolates XY between the last sample above the plane and the first
        sample at/below it, so the recorded capture point is comparable to the
        predicted intercept.  Runs only after release (not while holding).
        """
        if (
            self._actual_capture_m is not None
            or previous_pose is None
            or self.session.state is SessionState.WAIT_RELEASE
        ):
            return
        current_pose = self._last_raw_pose
        plane_z = self.config.prediction_plane_z_m
        fraction = descending_plane_crossing_fraction(
            previous_pose, current_pose, plane_z
        )
        if fraction is None:
            return
        crossing_xy = previous_pose[:2] + fraction * (current_pose[:2] - previous_pose[:2])
        self._actual_capture_m = np.array(
            [float(crossing_xy[0]), float(crossing_xy[1]), plane_z], dtype=float
        )
        if previous_time_ns is not None:
            self._actual_capture_time_ns = previous_time_ns + round(
                fraction * (current_time_ns - previous_time_ns)
            )

    def _clear_raw_measurements(self) -> None:
        """Drop stale raw samples so terminal rows do not show mismatched data."""
        self._last_raw_pose = None
        self._prev_raw_pose_time_ns = None

    def _apply_estimator_measurement(self, item: BufferedMeasurement) -> None:
        if item.source == "pose":
            result = self.estimator.update_pose(item.value, time_ns=item.time_ns)
            health = self._pose_health
        else:
            result = self.estimator.update_twist(item.value, time_ns=item.time_ns)
            health = self._twist_health
        health.record(
            receive_time_ns=item.receive_time_ns,
            accepted=result.accepted,
            update_reason=result.reason,
        )
        if not result.accepted and not result.reason.startswith("out_of_order"):
            self.get_logger().debug(f"{item.source} rejected: {result.reason}")
        if result.accepted and self._ekf_epoch_start_monotonic_s is None:
            self._ekf_epoch_start_monotonic_s = time.monotonic()

    def _on_payload_pose(self, message: PoseStamped) -> None:
        position = np.array(
            [message.pose.position.x, message.pose.position.y, message.pose.position.z],
            dtype=float,
        )
        if np.all(np.isfinite(position)):
            self._payload_position_m = position

    def _service_reply(self, response: Trigger.Response, accepted: bool, reason: str) -> Trigger.Response:
        response.success = bool(accepted)
        response.message = reason
        return response

    def _on_arm(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        transition = self.session.arm(monotonic_s=time.monotonic())
        if transition.accepted:
            self._reset_measurement_epoch()
            self.release_detector.reset()
            self.get_logger().info(
                "automatic release detector armed; waiting for ballistic free flight"
            )
        self._packet_reason = transition.reason
        return self._service_reply(response, transition.accepted, transition.reason)

    def _on_release(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        transition = self.session.release(monotonic_s=time.monotonic())
        if transition.accepted:
            self.release_detector.reset()
            self._reset_measurement_epoch(
                epoch_start_time_ns=int(self.get_clock().now().nanoseconds)
            )
            self.get_logger().info(
                "forced manual release accepted; EKF reset, waiting for new pose/twist"
            )
        self._packet_reason = transition.reason
        return self._service_reply(response, transition.accepted, transition.reason)

    def _activate_automatic_release(self, detection: FreeFlightDetection) -> None:
        transition = self.session.release(monotonic_s=time.monotonic())
        if not transition.accepted:
            self._packet_reason = transition.reason
            return
        self._reset_measurement_epoch(epoch_start_time_ns=detection.release_time_ns)
        for item in detection.measurements:
            self._apply_estimator_measurement(item)
        self._packet_reason = "free_flight_detected"
        latency_ms = (detection.confirmation_time_ns - detection.release_time_ns) * 1e-6
        self.get_logger().info(
            "free flight detected; "
            f"release_time_ns={detection.release_time_ns}, "
            f"latency_ms={latency_ms:.1f}, "
            f"position_rms_m={detection.position_rms_m:.4f}, "
            f"velocity_rms_mps={detection.velocity_rms_mps:.4f}, "
            f"acceleration_error_mps2={detection.acceleration_error_mps2:.3f}"
        )

    def _reset_measurement_epoch(self, *, epoch_start_time_ns: int | None = None) -> None:
        self.estimator.reset()
        self._pose_health.reset(epoch_start_time_ns=epoch_start_time_ns)
        self._twist_health.reset(epoch_start_time_ns=epoch_start_time_ns)
        self._last_processed_counts = (0, 0)
        self._latest_estimate = None
        self._latest_prediction = None
        self._latest_prediction_valid = False
        self._last_raw_pose = None
        self._prev_raw_pose_time_ns = None
        self._actual_capture_m = None
        self._actual_capture_time_ns = None
        self._ekf_epoch_start_monotonic_s = None

    def _on_rearm(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        transition = self.session.rearm(monotonic_s=time.monotonic())
        self._reset_measurement_epoch()
        self.release_detector.reset()
        self.mapper.reset()
        self._packet_reason = transition.reason
        self._create_object_subscriptions()
        return self._service_reply(response, transition.accepted, transition.reason)

    def _on_cancel(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        transition = self.session.cancel(monotonic_s=time.monotonic())
        self._latest_prediction_valid = False
        self._packet_reason = transition.reason
        if self.session.state is SessionState.DONE:
            self._destroy_object_subscriptions()
            self._clear_raw_measurements()
        return self._service_reply(response, transition.accepted, transition.reason)

    def _source_age_s(self, now_ns: int, source_time_ns: int | None) -> float:
        if source_time_ns is None:
            return math.inf
        return max(0.0, (now_ns - source_time_ns) * 1e-9)

    def _mark_lost(self, now_monotonic_s: float, reason: str) -> None:
        transition = self.session.mark_lost(
            monotonic_s=now_monotonic_s, payload_position_m=self._payload_position_m
        )
        self._latest_prediction_valid = False
        self._latest_prediction = None
        self._latest_estimate = None
        self._packet_reason = reason if transition.accepted else transition.reason

    def _end_session(self, now_monotonic_s: float, reason: str) -> None:
        transition = self.session.finish(monotonic_s=now_monotonic_s, reason=reason)
        self._latest_prediction_valid = False
        self._latest_prediction = None
        self._latest_estimate = None
        self._packet_reason = reason if transition.accepted else transition.reason

    def _process_new_estimate(self, now_ns: int, now_monotonic_s: float) -> bool:
        snapshot = self.estimator.snapshot()
        counts = (
            self.estimator.measurement_counts["pose"],
            self.estimator.measurement_counts["twist"],
        )
        if counts == self._last_processed_counts:
            return False
        self._last_processed_counts = counts
        if not snapshot.ready or snapshot.state_time_ns is None:
            self._mark_lost(now_monotonic_s, "estimator_not_ready")
            return True

        estimate_time_ns = max(now_ns, snapshot.state_time_ns)
        estimate = self.estimator.estimate_at(estimate_time_ns)
        prediction = predict_descending_crossing(
            estimate.state,
            estimate.covariance,
            state_time_ns=estimate_time_ns,
            plane_z_m=self.config.prediction_plane_z_m,
            gravity_mps2=self.config.gravity_mps2,
            drag_beta_m_inv=self.config.drag_beta_m_inv,
            max_step_s=self.config.rk4_max_step_s,
            max_time_s=self.config.ballistic_max_time_s,
        )
        if prediction is None:
            self._mark_lost(now_monotonic_s, "no_future_descending_crossing")
            return True

        mapping = self.mapper.consider(prediction.intercept_m)
        if not mapping.accepted:
            self._mark_lost(now_monotonic_s, mapping.reason)
            return True
        admission = self.session.admit_prediction(
            command_ready=mapping.command_ready,
            monotonic_s=now_monotonic_s,
            time_to_contact_s=prediction.time_to_contact_s,
        )
        self._latest_estimate = estimate
        self._latest_prediction = prediction
        self._latest_prediction_valid = admission.valid
        if not admission.accepted:
            self._packet_reason = mapping.reason if not mapping.command_ready else admission.reason
            return True
        self._packet_reason = mapping.reason if admission.valid else admission.reason
        return True

    def _packet(self) -> dict[str, object]:
        state = self.session.state
        valid = self._latest_prediction_valid and state is SessionState.TRACKING
        prediction = self._latest_prediction
        estimate = self._latest_estimate
        if prediction is None or estimate is None:
            intercept = [0.0, 0.0, self.config.prediction_plane_z_m]
            covariance = [0.0, 0.0, 0.0]
            radius = 0.0
            intercept_time_ns = 0
            time_to_contact_s = 0.0
            pose_time_ns = 0
            twist_time_ns = 0
            state_time_ns = 0
        else:
            intercept = prediction.intercept_m.tolist()
            covariance = prediction.cov_xy_m2.tolist()
            radius = prediction.xy_radius_95_m
            intercept_time_ns = prediction.intercept_time_ns
            time_to_contact_s = prediction.time_to_contact_s
            pose_time_ns = estimate.pose_time_ns or 0
            twist_time_ns = estimate.twist_time_ns or 0
            state_time_ns = estimate.state_time_ns or 0

        if valid:
            command_target = self.mapper.current_target_m
            confidence = confidence_from_radius(radius, self.config.usable_net_radius_m)
        elif state in (
            SessionState.LOST,
            SessionState.RECOVERING,
            SessionState.DONE,
            SessionState.ERROR,
        ):
            command_target = self.session.hold_target_m
            confidence = 0.0
        else:
            command_target = self.mapper.current_target_m
            confidence = 0.0
        hold_target = self.session.hold_target_m
        return {
            "schema": "netcatch.prediction.v1",
            "throw_id": self.session.throw_id,
            "state": state.value,
            "valid": valid,
            "frame_id": self.config.frame_id,
            "object_name": self.config.object_name,
            "reason": self._packet_reason or self.session.reason,
            "pose_time_ns": int(pose_time_ns),
            "twist_time_ns": int(twist_time_ns),
            "state_time_ns": int(state_time_ns),
            "intercept_time_ns": int(intercept_time_ns),
            "time_to_contact_s": float(time_to_contact_s),
            "prediction_plane_z_m": self.config.prediction_plane_z_m,
            "intercept_m": [float(value) for value in intercept],
            "command_target_m": [float(value) for value in command_target],
            "hold_target_m": [float(value) for value in hold_target],
            "cov_xy_m2": [float(value) for value in covariance],
            "xy_radius_95_m": float(radius),
            "confidence": float(confidence),
        }

    def _publish(self) -> None:
        packet = self._packet()
        if self.csv_logger is not None:
            self._log_csv_row(packet)
        message = String()
        message.data = encode_prediction_packet(packet)
        self._publisher.publish(message)

    def _log_csv_row(self, packet: dict[str, object]) -> None:
        """Append one debug row per published packet once the EKF epoch runs."""
        if self.csv_logger is None or self._ekf_epoch_start_monotonic_s is None:
            return
        now_monotonic_s = time.monotonic()
        raw_pose = self._last_raw_pose
        intercept = packet["intercept_m"]
        command = packet["command_target_m"]
        hold = packet["hold_target_m"]

        def triple(values: Sequence[float] | None, prefix: str) -> dict[str, object]:
            if values is None:
                return {f"{prefix}_x": "", f"{prefix}_y": "", f"{prefix}_z": ""}
            return {
                f"{prefix}_x": float(values[0]),
                f"{prefix}_y": float(values[1]),
                f"{prefix}_z": float(values[2]),
            }

        row: dict[str, object] = {
            "t_monotonic_s": now_monotonic_s,
            "ekf_elapsed_s": now_monotonic_s - self._ekf_epoch_start_monotonic_s,
            "throw_id": packet["throw_id"],
            "state": packet["state"],
            "valid": 1 if packet["valid"] else 0,
            "reason": packet["reason"],
        }
        row.update(triple(raw_pose, "pose_measured"))
        row.update(triple(intercept, "intercept"))
        row.update(
            {
                "time_to_contact_s": packet["time_to_contact_s"],
                "intercept_time_ns": packet["intercept_time_ns"],
                "xy_radius_95_m": packet["xy_radius_95_m"],
                "confidence": packet["confidence"],
            }
        )
        row.update(triple(command, "command_target"))
        row.update(triple(hold, "hold_target"))
        row.update(triple(self._actual_capture_m, "actual_capture"))
        row.update(
            {
                "actual_capture_time_ns": (
                    self._actual_capture_time_ns
                    if self._actual_capture_time_ns is not None
                    else ""
                ),
                "state_time_ns": packet["state_time_ns"],
            }
        )
        try:
            self.csv_logger.row(row)
        except Exception as exc:  # CSV failures must never take down prediction.
            self.get_logger().error(
                f"csv logging failed: {type(exc).__name__}: {exc}; disabling"
            )
            self.csv_logger.close()
            self.csv_logger = None

    def _on_publish_timer(self) -> None:
        now_monotonic_s = time.monotonic()
        now_ns = int(self.get_clock().now().nanoseconds)
        try:
            self.session.tick(monotonic_s=now_monotonic_s)
            if self.session.state is SessionState.DONE:
                self._latest_prediction_valid = False
                self._packet_reason = self.session.reason
                self._destroy_object_subscriptions()
                self._clear_raw_measurements()
                self._publish()
                return
            if self.session.state is SessionState.ERROR:
                self._latest_prediction_valid = False
                self._packet_reason = self.session.reason
                self._clear_raw_measurements()
                self._publish()
                return
            if self.session.state is SessionState.WAIT_RELEASE:
                self._latest_prediction_valid = False
                self._packet_reason = self.session.reason
                self._publish()
                return

            snapshot = self.estimator.snapshot()
            twist_age = self._source_age_s(now_ns, snapshot.twist_time_ns)
            pose_problem = self._pose_health.problem_reason(
                now_ns=now_ns,
                accepted_time_ns=snapshot.pose_time_ns,
                stale_after_s=self.config.lost_after_gap_s,
            )
            twist_problem = self._twist_health.problem_reason(
                now_ns=now_ns,
                accepted_time_ns=snapshot.twist_time_ns,
                stale_after_s=self.config.lost_after_gap_s,
            )
            if pose_problem is not None:
                self._end_session(now_monotonic_s, pose_problem)
            elif twist_problem is not None:
                self._end_session(now_monotonic_s, twist_problem)
            elif not snapshot.ready:
                self._latest_prediction_valid = False
                self._packet_reason = "waiting_for_post_release_measurements"
            else:
                processed = self._process_new_estimate(now_ns, now_monotonic_s)
                if (
                    processed
                    and self._latest_prediction_valid
                    and twist_age > self.config.twist_stale_s
                ):
                    self._packet_reason = "pose_only_grace"
            if self.session.state is SessionState.DONE:
                self._destroy_object_subscriptions()
                self._clear_raw_measurements()
            self._publish()
        except Exception as exc:  # ROS boundary: convert unexpected faults to finite ERROR packets.
            self.get_logger().error(f"prediction callback failed: {type(exc).__name__}: {exc}")
            if self.session.state not in (SessionState.DONE, SessionState.ERROR):
                self.session.fail(monotonic_s=now_monotonic_s, reason="prediction_callback_error")
            self._latest_prediction_valid = False
            self._latest_prediction = None
            self._latest_estimate = None
            self._packet_reason = self.session.reason
            self._publish()


def _parse_arguments(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="NetCatch NUC ballistic prediction process")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--csv",
        default=os.environ.get("NETCATCH_PREDICT_CSV"),
        help=(
            "Base path for CSV debug logging; a start-time stamp is inserted "
            "before the extension (e.g. logs/predict.csv -> "
            "logs/predict_20260902_101500_123456.csv). Empty disables logging. "
            "Defaults to $NETCATCH_PREDICT_CSV."
        ),
    )
    parsed, ros_args = parser.parse_known_args(argv)
    return parsed, ros_args


def main(argv: Sequence[str] | None = None) -> None:
    parsed, ros_args = _parse_arguments(argv)
    config = load_config(parsed.config)
    rclpy.init(args=ros_args)
    node = PredictionNode(config, csv_log_path=parsed.csv)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        if node.csv_logger is not None:
            node.csv_logger.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
