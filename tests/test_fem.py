"""Numerical regression tests of the P1 Galerkin/SUPG implementation."""
import numpy as np
import pytest

from advection_supg import exact_homogeneous, solve, stabilization_tau


@pytest.mark.parametrize("method", ["galerkin", "supg"])
def test_constant_dirichlet_solution(method):
    s = solve(n=17, method=method, left_value=2.5, right_value=2.5)
    np.testing.assert_allclose(s.phi, 2.5, rtol=0, atol=2e-12)
    assert s.relative_residual < 1e-12


@pytest.mark.parametrize("method", ["galerkin", "supg"])
def test_pure_diffusion_linear_solution(method):
    s = solve(n=33, method=method, velocity=0, epsilon=0.125,
              left_value=-1.2, right_value=2.8)
    np.testing.assert_allclose(s.phi, -1.2 + s.x, atol=5e-13)
    assert s.cell_peclet == 0
    assert s.tau == 0


def test_original_course_parameters_and_nodal_exactness():
    p = dict(n=20, length=4, velocity=1, epsilon=0.01, left_value=0, right_value=1)
    galerkin = solve(**p, method="galerkin")
    supg = solve(**p, method="supg")
    exact = exact_homogeneous(supg.x, **{k: v for k, v in p.items() if k != "n"})
    assert supg.cell_peclet == pytest.approx(10)
    assert np.max(np.abs(supg.phi - exact)) < 2e-12
    assert galerkin.phi.min() < -0.1  # nonphysical undershoot on this coarse mesh
    assert supg.phi.min() >= -1e-12
    assert supg.phi.max() <= 1 + 1e-12


def test_matches_historical_tridiagonal_stencil_galerkin():
    n, h, u, eps = 20, 4/20, 1., 0.01
    # Independently construct the centered stencil in assignment1_3.m.
    A = np.zeros((n + 1, n + 1))
    for j in range(1, n):
        A[j, j - 1] = -eps / h - u / 2
        A[j, j] = 2 * eps / h
        A[j, j + 1] = -eps / h + u / 2
    A[0, 0] = A[-1, -1] = 1
    b = np.zeros(n + 1)
    b[-1] = 1
    historical = np.linalg.solve(A, b)
    modern = solve(n=n, method="galerkin", length=4, velocity=u, epsilon=eps)
    np.testing.assert_allclose(modern.phi, historical, rtol=0, atol=1e-13)


def test_matches_historical_stabilized_stencil():
    n, h, u, eps = 20, .2, 1., .01
    pe = u * h / (2 * eps)
    eps_hat = 0.5 * u * h * (1 / np.tanh(pe) - 1 / pe)
    s = solve(n=n, method="supg", length=4, velocity=u, epsilon=eps)
    assert eps_hat == pytest.approx(s.tau * u**2)
    eps_eff = eps + eps_hat
    A = np.zeros((n + 1, n + 1))
    for j in range(1, n):
        A[j, j - 1] = -eps_eff / h - u / 2
        A[j, j] = 2 * eps_eff / h
        A[j, j + 1] = -eps_eff / h + u / 2
    A[0, 0] = A[-1, -1] = 1
    b = np.zeros(n + 1)
    b[-1] = 1
    np.testing.assert_allclose(s.phi, np.linalg.solve(A, b), atol=1e-13)


@pytest.mark.parametrize("velocity", [-1., -.5, 0., .5, 1.])
def test_exact_formula_matches_boundary_conditions(velocity):
    result = exact_homogeneous([0., 4.], velocity=velocity,
                               epsilon=.01, left_value=-3, right_value=7)
    np.testing.assert_allclose(result, [-3, 7], atol=5e-13, rtol=0)


@pytest.mark.parametrize("velocity", [-2., -1., .25, 1., 2.])
def test_supg_nodal_exactness_both_advection_directions(velocity):
    params = dict(n=30, length=1., velocity=velocity, epsilon=0.002,
                  left_value=-0.5, right_value=1.2)
    s = solve(method="supg", **params)
    exact = exact_homogeneous(s.x, **{k: v for k, v in params.items() if k != "n"})
    np.testing.assert_allclose(s.phi, exact, rtol=0, atol=3e-12)


def test_analytic_formula_no_overflow_extreme_peclet():
    xx = np.linspace(0, 1, 101)
    for velocity in (1000., -1000.):
        phi = exact_homogeneous(xx, length=1, velocity=velocity,
                                epsilon=0.1)
        assert np.all(np.isfinite(phi))
        assert phi.min() >= -1e-15 and phi.max() <= 1 + 1e-15


def test_tau_small_peclet_limit():
    h, eps = .1, .2
    tau = stabilization_tau(h=h, epsilon=eps, velocity=1e-10)
    assert tau == pytest.approx(h*h / (12 * eps), rel=1e-8)


def test_tau_large_peclet_limit():
    h, eps, velocity = .1, 1e-9, 1.
    assert stabilization_tau(h=h, epsilon=eps, velocity=velocity) == pytest.approx(h/(2*velocity), rel=1e-6)


def test_constant_source_pure_diffusion_quadratic_exact_nodes():
    # -epsilon phi'' = 2 epsilon for phi(x)=x(1-x), Dirichlet zero.
    for method in ("galerkin", "supg"):
        s = solve(n=30, length=1, epsilon=.3, velocity=0,
                  left_value=0, right_value=0, source=lambda x: .6,
                  method=method)
        np.testing.assert_allclose(s.phi, s.x*(1-s.x), atol=2e-14)


def test_linear_solution_with_nonzero_source():
    # phi(x)=2x-1, f=u phi' ; strongly advection-dominated.
    for method in ("galerkin", "supg"):
        s = solve(n=13, length=1, epsilon=0.001, velocity=2,
                  left_value=-1, right_value=1,
                  method=method, source=lambda x: np.full_like(x, 4.))
        np.testing.assert_allclose(s.phi, 2*s.x-1, atol=3e-13)


def test_smooth_manufactured_solution_second_order():
    u, eps = 1., .2
    source = lambda x: u*(1 + np.pi*np.cos(np.pi*x)) + eps*np.pi**2*np.sin(np.pi*x)
    results = {}
    for method in ("galerkin", "supg"):
        errs = []
        for n in (16, 32, 64):
            s = solve(n=n, length=1, velocity=u, epsilon=eps,
                      left_value=0, right_value=1, method=method, source=source)
            exact = s.x + np.sin(np.pi*s.x)
            errs.append(float(np.sqrt(np.mean((s.phi-exact)**2))))
            assert s.relative_residual < 1e-11
        rate = np.log2(errs[-2] / errs[-1])
        if method == "galerkin":
            assert 1.8 < rate < 2.2, (method, rate, errs)
        else:
            assert rate > 2.0, (method, rate, errs)
        results[method] = errs
    assert results["supg"][-1] < 1e-3


@pytest.mark.parametrize("bad", [0, -1, 2.3, True])
def test_invalid_n(bad):
    with pytest.raises(ValueError):
        solve(n=bad)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        solve(epsilon=0)
    with pytest.raises(ValueError):
        solve(method="unknown")
    with pytest.raises(ValueError):
        solve(source=lambda x: np.array([np.nan] * len(x)))
