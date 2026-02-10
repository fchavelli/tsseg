"""Simple binary node structure for merge-based segmentation algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False, frozen=False)
class Bnode:
    """Binary node describing a segment ``[start, end)``."""

    start: int
    end: int
    val: float
    left: "Bnode | None" = None
    right: "Bnode | None" = None
    parent: "Bnode | None" = None

    @property
    def gain(self) -> float:
        """Return the cost decrease obtained by splitting this node."""

        if self.left is None or self.right is None:
            return 0.0
        if np.isinf(self.val) and self.val < 0:
            return 0.0
        return float(self.val - (self.left.val + self.right.val))

    def __lt__(self, other: "Bnode") -> bool:
        return self.start < other.start
