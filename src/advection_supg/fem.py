"""Sparse P1 finite elements for steady one-dimensional advection–diffusion.

    -eps * phi'' + velocity * phi' = source,  x in (0, length),
    phi(0) = left_value, phi(length) = right_value.

SUPG uses a residual-based streamline term, with the exact fitted 1D tau.
Constant coefficients and a uniform mesh are assumed; the source may vary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

Method = Literal["galerkin", "supg"]
Array = NDArray[np.float64]


@dataclass(frozen=True)
class Solution:
    x: Array
    phi: Array
    method: str
    cell_peclet: float
    tau: float
    relative_residual: float


def stabilization_tau(*, h: float, velocity: float, epsilon: float) -> float:
    """Return 1D optimal tau = h/(2|u|) * (coth(Pe) - 1/Pe).

    Uses a power series near Pe=0 to avoid catastrophic cancellation.
    tau=0 at velocity=0 because the entire SUPG term then vanishes.
    """
    if not (np.isfinite(h) and h > 0 and np.isfinite(epsilon) and epsilon > 0
            and np.isfinite(velocity)):
        raise ValueError("Require finite h>0, epsilon>0 and finite velocity")
    speed = abs(velocity)
    if speed == 0.0:
        return 0.0
    pe = speed * h / (2.0 * epsilon)
    if pe < 1e-3:
        # coth(z)-1/z = z/3 - z^3/45 + 2z^5/945 + O(z^7)
        factor = pe * (1 / 3 - pe**2 / 45 + 2 * pe**4 / 945)
    else:
        factor = 1 / np.tanh(pe) - 1 / pe
    return float(h / (2 * speed) * factor)


def _evaluate_source(source: Callable[[Array], Array | float], pts: Array) -> Array:
    values = np.asarray(source(pts), dtype=float)
    if values.ndim == 0:
        values = np.full_like(pts, values.item())
    if values.shape != pts.shape or not np.all(np.isfinite(values)):
        raise ValueError("source must return finite scalar or an array matching input")
    return values


def solve(
    *, n: int = 20, length: float = 4.0, velocity: float = 1.0,
    epsilon: float = 0.01, left_value: float = 0.0,
    right_value: float = 1.0, method: Method = "supg",
    source: Callable[[Array], Array | float] | None = None,
) -> Solution:
    """Solve the boundary-value problem using uniform, continuous P1 FEM.

    The source is integrated by 4-point Gauss-Legendre quadrature. For the
    P1 trial space, the cellwise SUPG residual is u*phi_h' - source: the
    interior phi_h'' vanishes on each open element.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    if not (np.isfinite(length) and length > 0 and np.isfinite(epsilon)
            and epsilon > 0 and np.isfinite(velocity)
            and np.isfinite(left_value) and np.isfinite(right_value)):
        raise ValueError("length, epsilon must be positive and all inputs finite")
    if method not in ("galerkin", "supg"):
        raise ValueError("method must be 'galerkin' or 'supg'")
    h = length / n
    x = np.linspace(0.0, length, n + 1)
    pe = abs(velocity) * h / (2 * epsilon)
    tau = stabilization_tau(h=h, velocity=velocity, epsilon=epsilon) if method == "supg" else 0.0
    viscosity = epsilon + tau * velocity**2
    diffusion = (viscosity / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    advection = (velocity / 2.0) * np.array([[-1.0, 1.0], [-1.0, 1.0]])
    element_matrix = diffusion + advection
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n + 1)

    # Gauss quadrature on the reference interval [0, 1].
    z, w = np.polynomial.legendre.leggauss(4)
    z, w = (z + 1) / 2, w / 2
    shape = np.stack([1 - z, z], axis=1)
    dshape = np.array([-1.0 / h, 1.0 / h])

    for e in range(n):
        for i in range(2):
            for j in range(2):
                rows.append(e + i)
                cols.append(e + j)
                data.append(float(element_matrix[i, j]))
        if source is not None:
            f = _evaluate_source(source, x[e] + h * z)
            load = h * (shape.T @ (w * f))
            load += tau * velocity * dshape * h * np.dot(w, f)
            rhs[e:e + 2] += load

    matrix = coo_matrix((data, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    boundary = np.array([left_value, right_value])
    interior_matrix = matrix[1:-1, 1:-1].tocsc()
    interior_rhs = rhs[1:-1] - matrix[1:-1, [0, n]] @ boundary
    interior_phi = np.asarray(spsolve(interior_matrix, interior_rhs), dtype=float)
    phi = np.concatenate(([left_value], interior_phi, [right_value]))
    if not np.all(np.isfinite(phi)):
        raise ArithmeticError("non-finite numerical solution")
    residual = interior_matrix @ interior_phi - interior_rhs
    denom = max(np.linalg.norm(interior_rhs), np.linalg.norm(interior_matrix @ interior_phi), 1e-30)
    return Solution(x=x, phi=phi, method=method, cell_peclet=float(pe),
                    tau=float(tau), relative_residual=float(np.linalg.norm(residual) / denom))
