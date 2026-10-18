"""Validation of the electrode field solution and QPM coefficient."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwkit import electrodes as el

def test_parallel_plate_limit():
    d = 1e-6; Lam = 200e-6; V = 10.0
    x, Ey = el.field_map(Lam, d, V, gap=0.0, y=0.5e-6, nx=4096, nmax=801)
    mid = np.abs(Ey[np.argmin(np.abs(x - Lam / 4))])
    assert abs(mid - V / d) / (V / d) < 0.01, (mid, V / d)

def test_ideal_square_wave_qpm():
    Lam = 1.0; x = np.linspace(0, Lam, 4000, endpoint=False)
    chi = np.where(x < Lam / 2, 1.0, -1.0)
    assert abs(el.qpm_coefficient(x, chi, Lam) - 2 / np.pi) < 1e-3

def test_on_off_qpm():
    Lam = 1.0; x = np.linspace(0, Lam, 4000, endpoint=False)
    chi = np.where(x < Lam / 2, 1.0, 0.0)
    assert abs(el.qpm_coefficient(x, chi, Lam) - 1 / np.pi) < 1e-3

def test_array_field_matches_periodic_solution():
    """field_profile of an alternating electrode array equals the periodic Fourier solution (field_map) away from the
    array ends, and a wide electrode gives the parallel-plate field V/d."""
    d, y, Lam, V, gap = 5.3e-6, 2.6e-6, 50e-6, 30.0, 2e-6
    x, Ey = el.field_map(Lam, d, V, gap, y, nx=2048, nmax=401)
    nel = 80
    X = np.arange(-200e-6, nel * Lam / 2 + 200e-6, Lam / 2048)
    F = el.field_profile(X, el.top_potential_array(X, Lam / 2, [V if j % 2 == 0 else -V for j in range(nel)], gap), d, y)
    assert np.max(np.abs(np.interp(20 * Lam + x, X, F) + Ey)) / np.max(np.abs(Ey)) < 1e-6
    X = np.arange(-3e-3, 3e-3, 0.5e-6)
    F = el.field_profile(X, el.top_potential_array(X, 2e-3, [10.0], 0.0, x0=-1e-3), d, y)
    assert abs(F[len(X) // 2] * d / 10.0 - 1) < 1e-9
