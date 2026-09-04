#!/usr/bin/env python3
"""Analyze NetCatch prediction CSV debug logs (columns: see csv_logger.py)."""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

FILES = sorted(Path("predict/logs").glob("predict_*.csv"))


def load(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def f(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key, "")
    return None if raw == "" else float(raw)


def v3(row: dict[str, str], prefix: str) -> np.ndarray | None:
    vals = [f(row, f"{prefix}_{axis}") for axis in ("x", "y", "z")]
    return None if any(v is None for v in vals) else np.asarray(vals, dtype=float)


def analyze(path: Path) -> None:
    rows = load(path)
    print("=" * 78)
    print(f"FILE: {path.name}  ({len(rows)} rows)")
    print("=" * 78)
    if not rows:
        print("  (empty)")
        return

    t = np.asarray([f(r, "t_monotonic_s") for r in rows], dtype=float)
    elapsed = np.asarray([f(r, "ekf_elapsed_s") for r in rows], dtype=float)
    states = Counter(r["state"] for r in rows)
    reasons = Counter(r["reason"] for r in rows)
    valid_count = sum(1 for r in rows if r["valid"] == "1")

    print(f"  time span: {t[-1] - t[0]:.3f} s   ekf_elapsed: {elapsed[0]:.4f} -> {elapsed[-1]:.4f} s")
    print(f"  states: {dict(states)}   valid=true rows: {valid_count}")
    print("  top reasons:")
    for reason, count in reasons.most_common(10):
        print(f"    {reason:40s} {count:4d}")

    dt = np.diff(t)
    print(f"  dt median={np.median(dt)*1e3:.2f} ms  max gap={dt.max()*1e3:.1f} ms  "
          f"elapsed monotonic: {bool(np.all(np.diff(elapsed) > 0))}")

    # ---- per-state raw-vs-EKF residuals ----
    print("\n  per-state |raw - ekf| residuals:")
    for st in ("TRACKING", "LOST", "RECOVERING", "DONE"):
        sub = [r for r in rows if r["state"] == st]
        if not sub:
            continue
        perr, verr = [], []
        for r in sub:
            p, e = v3(r, "pose_measured"), v3(r, "ekf")
            tv, v = v3(r, "twist_measured"), _v3v(r)
            if p is not None and e is not None:
                perr.append(np.linalg.norm(p - e))
            if tv is not None and v is not None:
                verr.append(np.linalg.norm(tv - v))
        line = f"    {st:9s} n={len(sub):4d}"
        if perr:
            line += f"  |pose-ekf| mean={np.mean(perr):.4f} max={max(perr):.4f}"
        if verr:
            line += f"  |twist-ekfv| mean={np.mean(verr):.4f} max={max(verr):.4f}"
        print(line)

    # ---- predicted intercept vs actual capture ----
    print("\n  prediction vs actual capture (final values):")
    last_valid = None
    for r in reversed(rows):
        if r["valid"] == "1" and f(r, "intercept_x") is not None:
            last_valid = r
            break
    capture = None
    for r in reversed(rows):
        v = v3(r, "actual_capture")
        if v is not None and (v[0] != 0.0 or v[1] != 0.0):
            capture = v
            ctime = f(r, "actual_capture_time_ns")
            break
    if last_valid is not None:
        ix, iy = f(last_valid, "intercept_x"), f(last_valid, "intercept_y")
        ttc = f(last_valid, "time_to_contact_s")
        print(f"    last valid intercept: ({ix:.4f}, {iy:.4f})  ttc={ttc:.3f} s  "
              f"pose_z={f(last_valid, 'pose_measured_z'):.3f}")
    if capture is not None:
        err = np.linalg.norm(capture[:2] - np.array([ix, iy])) if last_valid is not None else float("nan")
        print(f"    actual capture point: ({capture[0]:.4f}, {capture[1]:.4f})  z={capture[2]:.3f}"
              + (f"   |intercept - actual|={err:.4f} m" if last_valid is not None else ""))
    else:
        print("    actual capture: (not latched; object did not cross plane)")

    # ---- ekf trajectory ----
    ekf_pos = np.vstack([v3(r, "ekf") for r in rows if v3(r, "ekf") is not None])
    if len(ekf_pos) > 1:
        dp = np.linalg.norm(np.diff(ekf_pos, axis=0), axis=1)
        print(f"\n  ekf position step: mean={dp.mean():.5f} m  max={dp.max():.5f} m")
    itc = [v3(r, "intercept") for r in rows if v3(r, "intercept") is not None and f(r, "intercept_x") != 0.0]
    if itc:
        itc = np.vstack(itc)
        print(f"  intercept XY over throw: x [{itc[:,0].min():.3f},{itc[:,0].max():.3f}]  "
              f"y [{itc[:,1].min():.3f},{itc[:,1].max():.3f}]")

    # ---- state_time sanity ----
    stn = [f(r, "state_time_ns") for r in rows if f(r, "state_time_ns") is not None]
    if len(stn) > 1 and stn[-1] != 0:
        print(f"  state_time_ns span: {(stn[-1]-stn[0])*1e-9:.3f} s")


def _v3v(row: dict[str, str]) -> np.ndarray | None:
    vals = [f(row, f"ekf_v{axis}") for axis in ("x", "y", "z")]
    return None if any(v is None for v in vals) else np.asarray(vals, dtype=float)


def main() -> None:
    paths = sys.argv[1:]
    for p in (paths or FILES):
        analyze(Path(p))


if __name__ == "__main__":
    main()
