"""Reproduce the historical n=20 advection-dominated comparison."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from advection_supg import solve, exact_homogeneous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/galerkin_vs_supg.png"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opts = dict(n=20, length=4, velocity=1, epsilon=0.01,
                left_value=0, right_value=1)
    standard = solve(method="galerkin", **opts)
    stabilized = solve(method="supg", **opts)
    dense_x = np.linspace(0, 4, 2000)
    exact = exact_homogeneous(dense_x, length=4, velocity=1, epsilon=0.01)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    for ax in axes:
        ax.plot(dense_x, exact, color="#18263A", lw=2.2, label="Analytical solution")
        ax.plot(standard.x, standard.phi, "o--", color="#DE6B48", ms=4,
                lw=1.5, label="Standard Galerkin")
        ax.plot(stabilized.x, stabilized.phi, "s-", color="#24816C", ms=4,
                lw=1.5, label="SUPG")
        ax.grid(True, alpha=.25)
        ax.set_xlabel("Position x")
        ax.set_ylabel("Transported scalar $\\phi(x)$")
    axes[0].set(title="Advection-dominated transport", xlim=(0, 4), ylim=(-1.12, 1.4))
    axes[1].set(title="Boundary layer at the outflow", xlim=(3.0, 4.02), ylim=(-1.12, 1.4))
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Galerkin vs SUPG   |   20 elements   |   cell Péclet = 10", fontsize=13)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("n=20, Pe_cell=10: Galerkin range", (float(standard.phi.min()), float(standard.phi.max())))
    print("SUPG maximum nodal error:", float(np.max(np.abs(
        stabilized.phi - exact_homogeneous(stabilized.x, length=4, velocity=1, epsilon=0.01)
    ))))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
