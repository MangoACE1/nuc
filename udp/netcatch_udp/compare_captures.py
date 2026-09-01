"""Compare JSONL multicast captures from multiple Jetson receivers."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from .protocol import WIRE_FIELDS


LOCAL_CAPTURE_FIELDS = frozenset({"sender", "receive_time_ns"})
EXPECTED_CAPTURE_FIELDS = WIRE_FIELDS | LOCAL_CAPTURE_FIELDS


class CaptureError(ValueError):
    """Raised when a receiver capture is not valid diagnostic JSONL."""


@dataclass(frozen=True, slots=True)
class CaptureSummary:
    name: str
    path: Path
    total_records: int
    unique_packets: int
    duplicate_seqs: tuple[int, ...]
    out_of_order_count: int
    missing_seqs: tuple[int, ...]
    longest_missing_run: int


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    passed: bool
    source_boot_id: str | None
    sender: str | None
    overlap_start_seq: int | None
    overlap_end_seq: int | None
    overlap_packets: int
    captures: tuple[CaptureSummary, ...]
    content_mismatch_seqs: tuple[int, ...]
    median_receive_skew_ms: float | None
    p95_receive_skew_ms: float | None
    max_receive_skew_ms: float | None
    errors: tuple[str, ...]


@dataclass(slots=True)
class _LoadedCapture:
    name: str
    path: Path
    records: list[dict[str, object]]
    packets_by_seq: dict[int, dict[str, object]]
    receive_time_by_seq: dict[int, int]
    boot_ids: set[str]
    senders: set[str]
    duplicate_seqs: set[int]
    out_of_order_count: int


def _require_nonnegative_int(name: str, value: object, *, location: str) -> int:
    if type(value) is not int or value < 0:
        raise CaptureError(f"{location}: {name} must be a nonnegative integer")
    return value


def _load_capture(name: str, path: Path) -> _LoadedCapture:
    if not path.is_file():
        raise CaptureError(f"{name}: capture is not readable: {path}")

    records: list[dict[str, object]] = []
    packets_by_seq: dict[int, dict[str, object]] = {}
    receive_time_by_seq: dict[int, int] = {}
    boot_ids: set[str] = set()
    senders: set[str] = set()
    duplicate_seqs: set[int] = set()
    out_of_order_count = 0
    previous_seq: int | None = None

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            location = f"{name}:{line_number}"
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise CaptureError(f"{location}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise CaptureError(f"{location}: root must be a JSON object")
            actual_fields = set(value)
            if actual_fields != EXPECTED_CAPTURE_FIELDS:
                missing = sorted(EXPECTED_CAPTURE_FIELDS - actual_fields)
                extra = sorted(actual_fields - EXPECTED_CAPTURE_FIELDS)
                raise CaptureError(
                    f"{location}: fields mismatch; missing={missing}, extra={extra}"
                )

            seq = _require_nonnegative_int("seq", value["seq"], location=location)
            receive_time_ns = _require_nonnegative_int(
                "receive_time_ns", value["receive_time_ns"], location=location
            )
            boot_id = value["source_boot_id"]
            sender = value["sender"]
            if not isinstance(boot_id, str) or not boot_id:
                raise CaptureError(f"{location}: source_boot_id must be a non-empty string")
            if not isinstance(sender, str) or not sender:
                raise CaptureError(f"{location}: sender must be a non-empty string")

            if previous_seq is not None and seq < previous_seq:
                out_of_order_count += 1
            previous_seq = seq
            if seq in packets_by_seq:
                duplicate_seqs.add(seq)
            else:
                packets_by_seq[seq] = {
                    field: value[field] for field in WIRE_FIELDS
                }
                receive_time_by_seq[seq] = receive_time_ns
            boot_ids.add(boot_id)
            senders.add(sender)
            records.append(value)

    if not records:
        raise CaptureError(f"{name}: capture has no JSON records: {path}")
    return _LoadedCapture(
        name=name,
        path=path,
        records=records,
        packets_by_seq=packets_by_seq,
        receive_time_by_seq=receive_time_by_seq,
        boot_ids=boot_ids,
        senders=senders,
        duplicate_seqs=duplicate_seqs,
        out_of_order_count=out_of_order_count,
    )


def _longest_consecutive_run(values: Sequence[int]) -> int:
    longest = 0
    current = 0
    previous: int | None = None
    for value in values:
        current = current + 1 if previous is not None and value == previous + 1 else 1
        longest = max(longest, current)
        previous = value
    return longest


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def compare_capture_files(
    captures: Mapping[str, Path | str],
    *,
    min_overlap_packets: int = 1,
    max_receive_skew_ms: float | None = None,
) -> ComparisonReport:
    if len(captures) < 2:
        raise CaptureError("at least two named captures are required")
    if type(min_overlap_packets) is not int or min_overlap_packets <= 0:
        raise CaptureError("min_overlap_packets must be a positive integer")
    if max_receive_skew_ms is not None and (
        isinstance(max_receive_skew_ms, bool)
        or not isinstance(max_receive_skew_ms, (int, float))
        or not math.isfinite(max_receive_skew_ms)
        or max_receive_skew_ms < 0.0
    ):
        raise CaptureError("max_receive_skew_ms must be finite and nonnegative")

    loaded = [
        _load_capture(name, Path(path))
        for name, path in sorted(captures.items())
    ]
    errors: list[str] = []

    for capture in loaded:
        if len(capture.boot_ids) != 1:
            errors.append(
                f"{capture.name}: expected one source_boot_id, got {sorted(capture.boot_ids)}"
            )
        if len(capture.senders) != 1:
            errors.append(f"{capture.name}: expected one sender, got {sorted(capture.senders)}")

    all_boot_ids = set().union(*(capture.boot_ids for capture in loaded))
    all_senders = set().union(*(capture.senders for capture in loaded))
    source_boot_id = next(iter(all_boot_ids)) if len(all_boot_ids) == 1 else None
    sender = next(iter(all_senders)) if len(all_senders) == 1 else None
    if source_boot_id is None:
        errors.append(f"captures do not share one source_boot_id: {sorted(all_boot_ids)}")
    if sender is None:
        errors.append(f"captures do not share one sender: {sorted(all_senders)}")

    starts = [min(capture.packets_by_seq) for capture in loaded]
    ends = [max(capture.packets_by_seq) for capture in loaded]
    overlap_start = max(starts)
    overlap_end = min(ends)
    overlap_packets = max(0, overlap_end - overlap_start + 1)
    if overlap_packets < min_overlap_packets:
        errors.append(
            f"overlap has {overlap_packets} packets, requires {min_overlap_packets}"
        )

    expected_seqs = (
        set(range(overlap_start, overlap_end + 1)) if overlap_packets > 0 else set()
    )
    summaries: list[CaptureSummary] = []
    for capture in loaded:
        missing = tuple(sorted(expected_seqs - set(capture.packets_by_seq)))
        summaries.append(
            CaptureSummary(
                name=capture.name,
                path=capture.path,
                total_records=len(capture.records),
                unique_packets=len(capture.packets_by_seq),
                duplicate_seqs=tuple(sorted(capture.duplicate_seqs)),
                out_of_order_count=capture.out_of_order_count,
                missing_seqs=missing,
                longest_missing_run=_longest_consecutive_run(missing),
            )
        )

    common_seqs = sorted(
        expected_seqs.intersection(
            *(set(capture.packets_by_seq) for capture in loaded)
        )
    )
    mismatch_seqs: list[int] = []
    receive_skews_ms: list[float] = []
    for seq in common_seqs:
        reference = loaded[0].packets_by_seq[seq]
        if any(capture.packets_by_seq[seq] != reference for capture in loaded[1:]):
            mismatch_seqs.append(seq)
        receive_times = [capture.receive_time_by_seq[seq] for capture in loaded]
        receive_skews_ms.append((max(receive_times) - min(receive_times)) / 1_000_000.0)

    median_skew = statistics.median(receive_skews_ms) if receive_skews_ms else None
    p95_skew = _nearest_rank(receive_skews_ms, 0.95) if receive_skews_ms else None
    maximum_skew = max(receive_skews_ms) if receive_skews_ms else None
    if (
        max_receive_skew_ms is not None
        and maximum_skew is not None
        and maximum_skew > max_receive_skew_ms
    ):
        errors.append(
            f"maximum receive skew {maximum_skew:.3f} ms exceeds "
            f"{float(max_receive_skew_ms):.3f} ms"
        )

    clean_captures = all(
        not summary.duplicate_seqs
        and summary.out_of_order_count == 0
        and not summary.missing_seqs
        for summary in summaries
    )
    passed = (
        not errors
        and overlap_packets >= min_overlap_packets
        and clean_captures
        and not mismatch_seqs
    )
    return ComparisonReport(
        passed=passed,
        source_boot_id=source_boot_id,
        sender=sender,
        overlap_start_seq=overlap_start if overlap_packets else None,
        overlap_end_seq=overlap_end if overlap_packets else None,
        overlap_packets=overlap_packets,
        captures=tuple(summaries),
        content_mismatch_seqs=tuple(mismatch_seqs),
        median_receive_skew_ms=median_skew,
        p95_receive_skew_ms=p95_skew,
        max_receive_skew_ms=maximum_skew,
        errors=tuple(errors),
    )


def _capture_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("must use NAME=/path/to/capture.jsonl")
    return name, Path(raw_path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare aligned multicast JSONL captures from multiple receivers"
    )
    parser.add_argument(
        "--capture",
        action="append",
        type=_capture_argument,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--min-overlap", type=_positive_int, default=1)
    parser.add_argument("--max-receive-skew-ms", type=_nonnegative_float)
    return parser


def _preview(values: Sequence[int], limit: int = 8) -> str:
    if not values:
        return "none"
    rendered = ",".join(str(value) for value in values[:limit])
    return rendered if len(values) <= limit else f"{rendered},..."


def render_report(report: ComparisonReport, *, stdout: TextIO) -> None:
    print(f"RESULT={'PASS' if report.passed else 'FAIL'}", file=stdout)
    print(f"source_boot_id={report.source_boot_id}", file=stdout)
    print(f"sender={report.sender}", file=stdout)
    print(
        f"overlap_seq={report.overlap_start_seq}..{report.overlap_end_seq} "
        f"packets={report.overlap_packets}",
        file=stdout,
    )
    print(
        "receive_skew_ms="
        f"median={report.median_receive_skew_ms} "
        f"p95={report.p95_receive_skew_ms} max={report.max_receive_skew_ms}",
        file=stdout,
    )
    print(
        f"content_mismatches={len(report.content_mismatch_seqs)} "
        f"seqs={_preview(report.content_mismatch_seqs)}",
        file=stdout,
    )
    for capture in report.captures:
        print(
            f"{capture.name}: records={capture.total_records} "
            f"unique={capture.unique_packets} "
            f"duplicates={len(capture.duplicate_seqs)} "
            f"out_of_order={capture.out_of_order_count} "
            f"missing={len(capture.missing_seqs)} "
            f"longest_missing_run={capture.longest_missing_run}",
            file=stdout,
        )
    for error in report.errors:
        print(f"error={error}", file=stdout)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    captures: dict[str, Path] = {}
    for name, path in args.capture:
        if name in captures:
            print(f"duplicate capture name: {name}", file=sys.stderr)
            return 2
        captures[name] = path
    try:
        report = compare_capture_files(
            captures,
            min_overlap_packets=args.min_overlap,
            max_receive_skew_ms=args.max_receive_skew_ms,
        )
    except CaptureError as exc:
        print(f"capture comparison failed: {exc}", file=sys.stderr)
        return 2
    render_report(report, stdout=sys.stdout)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
