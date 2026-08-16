"""Pre-registered family-wise error budget for ACRE experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StatisticalBudget:
    """One alpha ledger shared by synthesis, replay, mixture and validation."""

    delta_total: float = 0.05
    delta_synth: float | None = None
    delta_group: float | None = None
    delta_mix: float | None = None
    delta_validation: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < float(self.delta_total) < 1.0:
            raise ValueError("delta_total must be in (0, 1)")
        defaults = (self.delta_synth, self.delta_group, self.delta_mix, self.delta_validation)
        if any(value is not None and not 0.0 < float(value) < 1.0 for value in defaults):
            raise ValueError("budget components must be in (0, 1)")
        if sum(float(value if value is not None else self.delta_total / 4.0) for value in defaults) > float(self.delta_total) + 1e-12:
            raise ValueError("budget components exceed delta_total")

    @property
    def synth(self) -> float:
        return float(self.delta_synth if self.delta_synth is not None else self.delta_total / 4.0)

    @property
    def group(self) -> float:
        return float(self.delta_group if self.delta_group is not None else self.delta_total / 4.0)

    @property
    def mix(self) -> float:
        return float(self.delta_mix if self.delta_mix is not None else self.delta_total / 4.0)

    @property
    def validation(self) -> float:
        return float(self.delta_validation if self.delta_validation is not None else self.delta_total / 4.0)

    def group_delta(self, group_index: int) -> float:
        if group_index < 1:
            raise ValueError("group_index must be positive")
        return self.group / (group_index * (group_index + 1))

    def lattice_delta(self, lattice_size: int) -> float:
        if lattice_size < 1:
            raise ValueError("lattice_size must be positive")
        return self.synth / lattice_size

    def to_dict(self) -> dict[str, float]:
        return {
            "delta_total": float(self.delta_total),
            "delta_synth": self.synth,
            "delta_group": self.group,
            "delta_mix": self.mix,
            "delta_validation": self.validation,
        }


__all__ = ["StatisticalBudget"]
