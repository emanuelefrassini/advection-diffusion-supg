# Verification and reproducibility

This repository was constructed from two original MATLAB coursework files,
which are preserved unchanged under `legacy/coursework/`. The Python solver
is a **new**, independently tested implementation of the same one-dimensional
transport model and stabilized stencil.

## Zero-source course example

| Input | Value |
| --- | ---: |
| Domain length | 4 |
| Advection speed | 1 |
| Diffusion coefficient | 0.01 |
| Number of elements | 20 |
| Dirichlet values | 0, 1 |
| Cell Péclet number | 10 |

Verified with Python, NumPy and SciPy in November 2026:

- The standard Galerkin solution undershoots to approximately **−0.851644**.
- The fitted SUPG solution has maximum **nodal** error below **7e−17** against
  the analytical zero-source solution (on this machine).
- Tests independently recreate the original MATLAB tridiagonal Galerkin and
  stabilized stencil coefficients and verify that the Python solutions match.
- Numerical results use SciPy sparse assembly/solves rather than MATLAB.

## Nonzero-source manufactured case

For \(L=1\), \(u=1\), \(\varepsilon=0.2\), and Dirichlet values 0 and 1, use

\[
\phi_{\mathrm{exact}}(x)=x+\sin(\pi x),\quad
f(x)=1+\pi\cos(\pi x)+0.2\pi^2\sin(\pi x).
\]

This verifies the *source term* in both the Galerkin and SUPG forms. The
saved results in [`../results/manufactured_convergence.csv`](../results/manufactured_convergence.csv)
contain nodal RMSE for meshes with 8, 16, 32, 64 and 128 elements.

The finest-grid observed rates are about **1.994 for Galerkin** and
**3.994 for SUPG nodal error** in this specific smooth constant-coefficient
experiment. The unusually high observed SUPG rate is **not** a claim of
general fourth-order finite-element convergence, nor does it describe
continuous-domain norms or boundary-layer resolution.

## Tests

The `pytest` suite checks:

- zero-source constant and pure diffusion solutions;
- a pure diffusion quadratic manufactured solution;
- SUPG nodal exactness for positive and negative advection;
- matching the historical MATLAB tridiagonal stencils;
- the stabilisation parameter in small/large Péclet limits;
- finite and accurate analytical values at extreme Péclet numbers;
- nonzero-source linear and smooth manufactured solutions;
- input validation and small sparse linear residuals.

The suite passed **28 test cases** during preparation. GitHub Actions runs the
same tests under Python 3.11 and 3.13 once the repository is published.

## Limitations

- One spatial dimension, steady PDE and constant advective/diffusive
  coefficients; no time integration, finite-volume Navier–Stokes, turbulence,
  2D geometry or unstructured meshes.
- The exact nodal property applies only to the specified homogeneous 1D case.
- The historical MATLAB files were inspected but **not** executed as part of
  the Python test suite. Their source-dependent issues are documented in
  `legacy/README.md`; a third-party quadrature helper is not redistributed.
- The project is methodological and synthetic: no experimental or patient
  data, and no claim of external physical validation.
