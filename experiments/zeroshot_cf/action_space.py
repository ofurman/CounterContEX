"""Dependency-light action-space primitives for retained mixed-data search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OneHotActionGroup:
    """One categorical action represented by an atomic one-hot column group."""

    name: str
    columns: tuple[int, ...]
