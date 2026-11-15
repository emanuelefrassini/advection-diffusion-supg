"""Analytical zero-source 1D steady advection–diffusion solution."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def exact_homogeneous(
    x: ArrayLike, *, length: float = 4.0, velocity: float = 1.0,
    epsilon: float = 0.01, left_value: float = 0.0, right_value: float = 1.0,
) -> NDArray[np.float64]:
    """Evaluate the constant-coefficient zero-source solution stably.

    Avoids overflow for large positive velocity*length/epsilon and handles
    zero or negative advection correctly.
    """
    if not (np.isfinite(length) and length > 0 and np.isfinite(epsilon)
            and epsilon > 0 and np.isfinite(velocity)):
        raise ValueError("Require length>0, epsilon>0 and finite velocity")
    xx = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(xx)):
        raise ValueError("x must be finite")
    peclet_domain = velocity * length / epsilon
    t = xx / length
    if abs(peclet_domain) < 1e-8:
        ratio = t + 0.5 * peclet_domain * t * (t - 1)
    elif peclet_domain > 50:
        # exp(p*t)-1 / exp(p)-1, divided through by exp(p).
        ratio = np.exp(peclet_domain * (t - 1)) * (
            -np.expm1(-peclet_domain * t)
        ) / (-np.expm1(-peclet_domain))
    else:
        ratio = np.expm1(peclet_domain * t) / np.expm1(peclet_domain)
    return np.asarray(left_value + (right_value - left_value) * ratio)
