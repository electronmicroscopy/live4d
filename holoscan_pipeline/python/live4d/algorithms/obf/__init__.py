"""Optimum bright field (OBF) direct ptychography, a live channel next to parallax.

`core` is the method: the correlation route of quantem's `DirectPtychographyMontage`, ported to
torch on the parallax geometry (`ObfSetup`: the probe, the canvas layout and the Gram;
`reconstruct_once`: a whole `latest` store in one call). It imports no Holoscan; the parallax
lifecycle (`live4d.parallax.obf`) steps it inside the operator callbacks.
"""

from live4d.algorithms.obf.core import ObfSetup, reconstruct_once

__all__ = ["ObfSetup", "reconstruct_once"]
