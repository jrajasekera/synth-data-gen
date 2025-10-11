"""Utility helpers for RNG and JSON processing."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass(slots=True)
class RNGConfig:
    """Configuration for deterministic random number generation."""

    seed: Optional[int] = None

    def fork(self, offset: int) -> "RNGConfig":
        """Return a new config with a deterministic offset applied."""

        if self.seed is None:
            return RNGConfig(None)
        return RNGConfig(self.seed + offset)

    def random(self) -> random.Random:
        """Create a `random.Random` instance."""

        return random.Random(self.seed)

    def numpy(self) -> np.random.Generator:
        """Create a numpy Generator seeded consistently."""

        return np.random.default_rng(self.seed)


def stable_hash(values: Iterable[object]) -> str:
    """Generate a stable SHA-256 hash for a sequence of values."""

    serialized = json.dumps(list(values), sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
