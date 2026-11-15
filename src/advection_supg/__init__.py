"""1D Galerkin finite elements and streamline-upwind/Petrov–Galerkin (SUPG)."""
from .analytic import exact_homogeneous
from .fem import Solution, solve, stabilization_tau

__all__ = ["Solution", "solve", "stabilization_tau", "exact_homogeneous"]
