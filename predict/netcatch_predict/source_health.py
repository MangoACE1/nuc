"""Explain why a VRPN measurement source is currently unusable."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class SourceHealth:
    """Track callback receipt separately from estimator acceptance."""

    source: str
    last_receive_time_ns: int | None = field(init=False, default=None)
    last_update_accepted: bool | None = field(init=False, default=None)
    last_update_reason: str | None = field(init=False, default=None)
    epoch_start_time_ns: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")

    def reset(self, *, epoch_start_time_ns: int | None = None) -> None:
        if epoch_start_time_ns is not None and (
            isinstance(epoch_start_time_ns, bool)
            or not isinstance(epoch_start_time_ns, int)
            or epoch_start_time_ns < 0
        ):
            raise ValueError("epoch_start_time_ns must be None or a nonnegative integer")
        self.last_receive_time_ns = None
        self.last_update_accepted = None
        self.last_update_reason = None
        self.epoch_start_time_ns = epoch_start_time_ns

    def record(
        self,
        *,
        receive_time_ns: int,
        accepted: bool,
        update_reason: str,
    ) -> None:
        if isinstance(receive_time_ns, bool) or not isinstance(receive_time_ns, int):
            raise ValueError("receive_time_ns must be a nonnegative integer")
        if receive_time_ns < 0:
            raise ValueError("receive_time_ns must be a nonnegative integer")
        if type(accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        if not isinstance(update_reason, str) or not update_reason:
            raise ValueError("update_reason must be a non-empty string")
        self.last_receive_time_ns = receive_time_ns
        self.last_update_accepted = accepted
        self.last_update_reason = update_reason

    @staticmethod
    def _age_s(now_ns: int, sample_time_ns: int | None) -> float:
        if sample_time_ns is None:
            return math.inf
        return max(0.0, (now_ns - sample_time_ns) * 1e-9)

    def problem_reason(
        self,
        *,
        now_ns: int,
        accepted_time_ns: int | None,
        stale_after_s: float,
    ) -> str | None:
        if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative integer")
        if accepted_time_ns is not None and (
            isinstance(accepted_time_ns, bool)
            or not isinstance(accepted_time_ns, int)
            or accepted_time_ns < 0
        ):
            raise ValueError("accepted_time_ns must be None or a nonnegative integer")
        if not isinstance(stale_after_s, (int, float)) or isinstance(stale_after_s, bool):
            raise ValueError("stale_after_s must be a finite positive number")
        if not math.isfinite(stale_after_s) or stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be a finite positive number")

        if self._age_s(now_ns, accepted_time_ns) <= stale_after_s:
            return None
        if (
            accepted_time_ns is None
            and self.epoch_start_time_ns is not None
            and self._age_s(now_ns, self.epoch_start_time_ns) <= stale_after_s
        ):
            return None
        if self.last_receive_time_ns is None:
            return f"{self.source}_not_received"
        if self._age_s(now_ns, self.last_receive_time_ns) > stale_after_s:
            return f"{self.source}_stale"
        if self.last_update_accepted is False and self.last_update_reason is not None:
            reason = self.last_update_reason
            source_prefix = f"{self.source}_"
            if reason.startswith(source_prefix):
                reason = reason[len(source_prefix) :]
            return f"{self.source}_rejected_{reason}"
        return f"{self.source}_not_accepted"
