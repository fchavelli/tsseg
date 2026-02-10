"""Custom exceptions used by the vendored ruptures components."""


class NotEnoughPoints(Exception):
    """Raised when a segment is too short for a cost evaluation."""


class BadSegmentationParameters(Exception):
    """Raised when segmentation parameters admit no feasible solution."""
