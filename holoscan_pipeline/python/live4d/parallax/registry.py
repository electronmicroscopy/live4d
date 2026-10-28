"""The `--algorithm` name → `Algorithm` map.

Adding a method: implement `Accumulator` + `Algorithm` under `processing/<method>/`, import its
`Algorithm` here, and add one entry. `live4d.parallax.offline` and the ingest graph need no changes.
"""

from __future__ import annotations

from live4d.algorithms.base import Algorithm
from live4d.parallax.algorithm import ParallaxAlgorithm

ALGORITHMS: dict[str, Algorithm] = {
    ParallaxAlgorithm.name: ParallaxAlgorithm(),
}


def names() -> list[str]:
    """Registered algorithm names, for the `--algorithm` help/choices."""
    return sorted(ALGORITHMS)


def get(name: str) -> Algorithm:
    """Look up a registered algorithm, or exit with a helpful message."""
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise SystemExit(
            f"unknown algorithm {name!r}; available: {', '.join(names())}"
        )
