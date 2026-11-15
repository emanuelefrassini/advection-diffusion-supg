"""Convergence on a smooth manufactured solution with a nonzero source."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from advection_supg import solve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/manufactured_convergence.png"))
    parser.add_argument("--csv", type=Path, default=Path("results/manufactured_convergence.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    length, velocity, epsilon = 1., 1., 0.2
    exact = lambda x: x + np.sin(np.pi * x)
    source = lambda x: velocity * (1 + np.pi * np.cos(np.pi * x)) + epsilon * np.pi**2 * np.sin(np.pi * x)
    rows = []
    for n in (8, 16, 32, 64, 128):
        for method in ("galerkin", "supg"):
            sol = solve(n=n, length=length, velocity=velocity, epsilon=epsilon,
                        left_value=0., right_value=1., method=method, source=source)
            error = float(np.sqrt(np.mean((sol.phi - exact(sol.x))**2)))
            rows.append(dict(n=n, h=length/n, method=method, nodal_rmse=error,
                             cell_peclet=sol.cell_peclet))
            print(f"n={n:3d} method={method:8s} nodal RMSE={error:.6e}")
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for method, color, marker in (("galerkin", "#DE6B48", "o"), ("supg", "#24816C", "s")):
        sub = [row for row in rows if row["method"] == method]
        h = np.array([row["h"] for row in sub])
        e = np.array([row["nodal_rmse"] for row in sub])
        ax.loglog(h, e, marker + "-", color=color, lw=2, label=method.title())
        rate = np.log2(e[-2] / e[-1])
        print(f"Observed finest-grid {method} rate: {rate:.3f}")
    h = np.array([row["h"] for row in rows if row["method"] == "galerkin"])
    ax.loglog(h, 0.65 * h**2, "k:", lw=1.5, label="$O(h^2)$ guide")
    ax.set(xlabel="Element size h", ylabel="Nodal root-mean-square error",
           title="Manufactured-solution convergence (nonzero source)")
    ax.grid(True, which="both", alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", args.output, "and", args.csv)


if __name__ == "__main__":
    main()
