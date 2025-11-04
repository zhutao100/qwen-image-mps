from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

_DIMENSION_PATTERN = re.compile(r"\d+")


def _coerce_ints(values: Iterable[str]) -> Tuple[int, int]:
    ints = []
    for raw in values:
        raw = raw.strip()
        if not raw:
            continue
        try:
            number = int(raw)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid dimension component: '{raw}'") from exc
        if number <= 0:
            raise ValueError("Dimensions must be positive integers")
        ints.append(number)
    if len(ints) != 2:
        raise ValueError("Expected two positive integers for dimensions (width and height)")
    return ints[0], ints[1]


def parse_dimensions(value: Optional[object]) -> Optional[Tuple[int, int]]:
    """Parse fuzzy dimension input into an explicit (width, height) tuple.

    Accepts forms such as "1920x1080", "1920,1080", "[1920, 1080]", "(1920 1080)" or
    any other representation containing exactly two integers."""

    if value is None:
        return None

    if isinstance(value, (tuple, list)):
        return _coerce_ints(str(v) for v in value)

    text = str(value).strip()
    if not text:
        return None

    matches = _DIMENSION_PATTERN.findall(text)
    if len(matches) != 2:
        raise ValueError(
            "Could not parse dimensions. Provide exactly two integers, e.g. '1920x1080'."
        )
    return _coerce_ints(matches)
