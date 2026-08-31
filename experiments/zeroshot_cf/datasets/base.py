"""Provider contract and portable dataset configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from experiments.zeroshot_cf.core.contracts import PreparedDataset


@dataclass(frozen=True)
class DatasetSpec:
    """Scientific inputs that control dataset preparation."""

    name: str
    validation_fraction: float = 0.0
    drop_heloc_all_minus9: bool = False
    split_seed: int = 42

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.split_seed < 0:
            raise ValueError("split_seed must be non-negative")


class DatasetProvider(Protocol):
    """Prepare provider-neutral arrays and metadata from a dataset spec."""

    def prepare(self, spec: DatasetSpec) -> PreparedDataset: ...
