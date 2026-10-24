"""Parallax / tcBF — the worked example algorithm.

`engine` is the method (geometry + accumulator); `config` holds the tunable parameters and the
shared CLI knobs; `algorithm` wires it into the runner behind the `Algorithm` contract.
"""

from realtime_stem.processing.parallax.engine import (
    ParallaxAccumulator,
    ParallaxGeometry,
    preprocess_geometry,
)

__all__ = ["ParallaxAccumulator", "ParallaxGeometry", "preprocess_geometry"]
