"""Reference data for u_t + u*u_x = nu*u_xx on the periodic unit interval.

Uses the Cole-Hopf transformation with a Galilean shift for nonzero means.
The formula is continuous; its FFT evaluation has spatial discretization error.
This code generates labels only. It is never called inside the FNO forward pass.
"""

import math
import numpy as np


def sample_coefficients(count, seed, harmonics=8):
    rng = np.random.default_rng(seed)
    k = np.arange(1, harmonics + 1, dtype=np.float64)
    coefficients = rng.normal(size=(count, 2, harmonics)) / k**2
    amplitude = rng.uniform(0.2, 1.0, size=(count, 1, 1))
    coefficients *= amplitude / np.abs(coefficients).sum(axis=(1, 2), keepdims=True)
    mean = rng.uniform(-0.3, 0.3, size=(count, 1))
    return coefficients, mean


def initial_profile(coefficients, mean, n):
    k = np.arange(1, coefficients.shape[-1] + 1, dtype=np.float64)
    angles = 2 * np.pi * k[:, None] * np.arange(n)[None, :] / n
    return mean + coefficients[:, 0] @ np.sin(angles) + coefficients[:, 1] @ np.cos(angles)


def solve_burgers(coefficients, mean, n=512, viscosity=0.01, final_time=1.0):
    """Return float64 solutions for finite Fourier-series initial conditions.

    Let m = mean(u0), v0 = u0-m, and P_x = v0.
    phi0 = exp(-P/(2*nu)); phi_t = nu*phi_xx.
    Then u(x,t) = m - 2*nu * phi_x(x-m*t,t) / phi(x-m*t,t).
    The Galilean shift is essential when m != 0.
    """
    if viscosity <= 0 or final_time < 0 or n <= 2 * coefficients.shape[-1]:
        raise ValueError('Require nu > 0, T >= 0, and a grid resolving all input harmonics')
    if final_time == 0:
        return initial_profile(coefficients, mean, n)
    k = np.arange(1, coefficients.shape[-1] + 1, dtype=np.float64)
    omega = 2 * np.pi * k
    angles = omega[:, None] * np.arange(n)[None, :] / n
    primitive = -(coefficients[:, 0] / omega) @ np.cos(angles)
    primitive += (coefficients[:, 1] / omega) @ np.sin(angles)
    exponent = -primitive / (2 * viscosity)
    # Multiplying phi by a positive constant does not change phi_x/phi.
    exponent -= exponent.max(axis=-1, keepdims=True)
    phi_initial = np.exp(exponent)
    frequencies = 2 * np.pi * np.fft.rfftfreq(n, d=1 / n)
    evolution = np.exp(-viscosity * frequencies**2 * final_time)
    shift = np.exp(-1j * mean * frequencies[None, :] * final_time)
    phi_hat = np.fft.rfft(phi_initial, axis=-1) * evolution * shift
    phi = np.fft.irfft(phi_hat, n=n, axis=-1)
    phi_x = np.fft.irfft(1j * frequencies * phi_hat, n=n, axis=-1)
    if not np.isfinite(phi).all() or np.any(phi <= 1e-14 * phi.max(axis=-1, keepdims=True)):
        raise FloatingPointError('Cole-Hopf reconstruction is poorly conditioned for these settings')
    return mean - 2 * viscosity * phi_x / phi


def sample_burgers_pairs(count, n, seed, solver_n=512):
    """Fixed nu=0.01, T=1. Return (B,N,1) float32 training arrays."""
    if solver_n % n != 0:
        raise ValueError('The reference grid must be a multiple of the model grid')
    coefficients, mean = sample_coefficients(count, seed)
    u0 = initial_profile(coefficients, mean, n)
    target_fine = solve_burgers(coefficients, mean, n=solver_n)
    target = target_fine[:, ::solver_n // n]
    return u0[..., None].astype(np.float32), target[..., None].astype(np.float32)


def independent_rk4(u0, viscosity=0.01, final_time=1.0):
    """Small verification solver: dealiased Fourier differentiation plus RK4.

    This does not use the Cole-Hopf transformation. It is only for checking
    the reference targets, not for generating the full training dataset.
    """
    n = u0.shape[-1]
    modes = np.fft.rfftfreq(n, d=1 / n)
    omega = 2 * np.pi * modes
    keep = modes < n / 3
    largest_omega = omega[keep].max()
    max_dt = min(0.5 / (viscosity * largest_omega**2), 0.25 / (largest_omega * max(1, np.abs(u0).max())))
    steps = math.ceil(final_time / max_dt)
    dt = final_time / steps
    state = np.fft.rfft(u0, axis=-1) * keep

    def rhs(spectrum):
        u = np.fft.irfft(spectrum * keep, n=n, axis=-1)
        nonlinear = -0.5j * omega * np.fft.rfft(u * u, axis=-1)
        return (nonlinear - viscosity * omega**2 * spectrum) * keep

    for _ in range(steps):
        a = rhs(state)
        b = rhs(state + 0.5 * dt * a)
        c = rhs(state + 0.5 * dt * b)
        d = rhs(state + dt * c)
        state += dt * (a + 2*b + 2*c + d) / 6
    return np.fft.irfft(state, n=n, axis=-1)


def check_reference():
    coefficients, mean = sample_coefficients(16, seed=91)
    initial = initial_profile(coefficients, mean, 512)
    target = solve_burgers(coefficients, mean, n=512)
    refined = solve_burgers(coefficients, mean, n=1024)[:, ::2]
    refinement_error = np.max(np.abs(target - refined))
    assert refinement_error < 1e-8, refinement_error
    mass_error = np.max(np.abs(target.mean(-1, keepdims=True) - mean))
    assert mass_error < 1e-10, mass_error
    assert np.all(np.mean(target**2, axis=-1) <= np.mean(initial**2, axis=-1) + 1e-12)
    np.testing.assert_allclose(solve_burgers(coefficients, mean, final_time=0), initial)
    constants = solve_burgers(np.zeros((2, 2, 8)), np.array([[0.2], [-0.3]]))
    np.testing.assert_allclose(constants, np.broadcast_to([[0.2], [-0.3]], constants.shape), atol=1e-12)

    # Compare an independent method at the full requested final time.
    u0 = initial_profile(coefficients[:3], mean[:3], 256)
    numerical = independent_rk4(u0)
    reference = solve_burgers(coefficients[:3], mean[:3], n=256)
    relative_error = np.linalg.norm(numerical-reference, axis=-1) / np.linalg.norm(reference, axis=-1)
    assert relative_error.max() < 1e-5, relative_error
    result = {
        'max_abs_grid_refinement_512_vs_1024': float(refinement_error),
        'max_abs_mass_error': float(mass_error),
        'max_relative_l2_vs_independent_rk4': float(relative_error.max()),
        'checks': 'T=0, constants, mass conservation, energy dissipation, refinement, independent RK4 at T=1',
    }
    print('Burgers reference checks passed:', result, flush=True)
    return result


if __name__ == '__main__':
    check_reference()
