"""Response-surface registry. Each surface has a known global optimum."""

from __future__ import annotations

from tuneforge.surfaces.ackley import Ackley
from tuneforge.surfaces.base import Surface
from tuneforge.surfaces.branin import Branin
from tuneforge.surfaces.null_surface import NullSurface

SURFACES = {
    "branin": Branin,
    "ackley": Ackley,
    "null": NullSurface,
}


def make_surface(name: str) -> Surface:
    key = name.strip().lower()
    if key not in SURFACES:
        raise ValueError(f"unknown surface {name!r}; choose from {sorted(SURFACES)}")
    return SURFACES[key]()


__all__ = ["Surface", "SURFACES", "make_surface", "Branin", "Ackley", "NullSurface"]
