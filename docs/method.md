# Mathematical formulation

## Model

The code solves the **one-dimensional**, steady, constant-coefficient transport problem

\[
-\varepsilon\phi''(x) + u\phi'(x) = f(x),\qquad 0<x<L,\qquad
\phi(0)=\phi_L,\quad \phi(L)=\phi_R.
\]

Here \(u\) is a prescribed *signed* advective velocity and \(\varepsilon>0\) is a
positive diffusion coefficient. This project is a numerical PDE model problem,
**not** a two-dimensional CFD or Navier–Stokes solver.

## Standard Galerkin FEM

On a uniform mesh with \(n\) intervals and \(h=L/n\), continuous piecewise
linear (\(P_1\)) trial and test functions yield

\[
\varepsilon\int_0^L w_h'\phi_h'\,dx
+u\int_0^L w_h\phi_h'\,dx
=\int_0^L w_h f\,dx.
\]

Using the local nodal basis \(N_1=1-\xi\), \(N_2=\xi\), an element contributes

\[
K_e^{\mathrm{G}}=
\frac{\varepsilon}{h}\begin{pmatrix}1&-1\\-1&1\end{pmatrix}
+\frac{u}{2}\begin{pmatrix}-1&1\\-1&1\end{pmatrix}.
\]

Dirichlet data are imposed by eliminating both boundary unknowns from the
interior linear system, and the sparse matrix is solved with SciPy `spsolve`.

## SUPG stabilization

The residual-based streamline term is

\[
\sum_e\int_{K_e}\tau u w'_h
\underbrace{\left(-\varepsilon\phi_h''+u\phi'_h-f\right)}_{\text{strong residual}}\,dx.
\]

For piecewise-linear trial functions, \(\phi_h''=0\) **inside** each element,
so the assembled matrix gains

\[
K_e^{\mathrm{SUPG}}=K_e^{\mathrm{G}}
+\frac{\tau u^2}{h}
\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.
\]

The RHS gains \(\tau u \int_{K_e} N_i' f\,dx\). Both the Galerkin and SUPG
source loads are integrated using four-point Gauss–Legendre quadrature.

For constant coefficients, uniform \(P_1\) elements and the classical
one-dimensional optimal parameter,

\[
\mathrm{Pe}_h=\frac{|u|h}{2\varepsilon},\qquad
\tau=\frac{h}{2|u|}\left(\coth(\mathrm{Pe}_h)-\frac{1}{\mathrm{Pe}_h}\right).
\]

The implementation uses a Taylor expansion for small \(\mathrm{Pe}_h\) to avoid
floating-point cancellation. For zero velocity, the SUPG contribution is zero.
In **one dimension**, this stabilization is equivalent to an added diffusion
\(\varepsilon_{\mathrm{art}}=\tau u^2\). This equivalence should not be
interpreted as a general description of multidimensional SUPG.

## Analytical homogeneous reference

For \(f=0\), the exact solution is

\[
\phi(x)=\phi_L+(\phi_R-\phi_L)
\frac{e^{ux/\varepsilon}-1}{e^{uL/\varepsilon}-1},\qquad u\ne0.
\]

For \(u=0\), this becomes linear interpolation between the boundary data.
The implementation evaluates the exponential quotient using rescaling and
`expm1` to avoid overflow and cancellation for large Péclet numbers.

The classical SUPG parameter gives a **nodally exact** solution for this
particular *homogeneous*, constant-coefficient uniform-mesh problem. Nodal
exactness does **not** mean that a coarse \(P_1\) interpolant is accurate
between nodes in a thin outflow layer.

## References

- R. Codina (1998), *Comparison of some finite element methods for solving the
  diffusion-convection-reaction equation*, Computer Methods in Applied
  Mechanics and Engineering 156:185–210. [Open PDF](https://deca.upc.edu/en/people/ramon.codina/meflu/art-prac2.pdf)
- [SciPy sparse direct solve (`spsolve`)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html)
- Original user-provided MATLAB coursework scripts in [`../legacy/coursework/`](../legacy/coursework/).
