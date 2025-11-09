"""
Configuration helpers for selector21.

Provides lightweight dataclasses so that gating and recency parameters travel
through the codebase in a structured way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GateConfig:
    cvd_slope_min: float = 0.0
    imbalance_min: float = 0.0
    atr_z_min: float = 0.0
    vhf_min: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "cvd_slope_min": float(self.cvd_slope_min),
            "imbalance_min": float(self.imbalance_min),
            "atr_z_min": float(self.atr_z_min),
            "vhf_min": float(self.vhf_min),
        }


@dataclass(frozen=True)
class RecencyConfig:
    mode: str = ""
    half_life: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return bool(self.mode and (self.half_life or 0) > 0)


def gate_config_from_args(args) -> GateConfig:
    """Build GateConfig from an argparse.Namespace-like object."""
    return GateConfig(
        cvd_slope_min=float(getattr(args, "cvd_slope_min", 0.0) or 0.0),
        imbalance_min=float(getattr(args, "imbalance_min", 0.0) or 0.0),
        atr_z_min=float(getattr(args, "atr_z_min", 0.0) or 0.0),
        vhf_min=float(getattr(args, "vhf_min", 0.0) or 0.0),
    )


def recency_config_from_args(args) -> RecencyConfig:
    return RecencyConfig(
        mode=str(getattr(args, "ml_recency_mode", "") or "").strip().lower(),
        half_life=float(getattr(args, "ml_recency_half_life", 0) or 0) or None,
    )


__all__ = ["GateConfig", "RecencyConfig", "gate_config_from_args", "recency_config_from_args"]
