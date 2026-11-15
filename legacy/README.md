# Original MATLAB coursework — historical source

These two `.m` files were copied **unchanged** from my CFD coursework archive.
They are retained for provenance and are **not** the validated Python implementation.

- `assignment1_3.m`: uses `L=4`, `u=1`, `epsilon=0.01`, `n=20`,
  Dirichlet values `0,1`, and a zero source. It compares centered Galerkin
  against a stabilization via fitted additional diffusion. Its plot labels
  a **1000-element numerical Galerkin solution** as `"Exact"`; it is not an
  analytical exact solution.
- `advection_diffusion_fem.m`: constructs a sparse tridiagonal 1D operator,
  with optional source integration. The `fem_flag=false` RHS branch assigns
  only one component due to the loop index being reused; this is a known
  historical bug. The `fem_flag=true` branch depends on third-party `lgwt.m`,
  which is **not redistributed** in this repository.

The Python solver reproduces the same constant-coefficient 1D differential
problem and compares FEM against an analytical zero-source reference. It is a
**new implementation**, not a line-by-line port or a claim that the legacy
source was bug-free. Historical figures have been regenerated independently.
