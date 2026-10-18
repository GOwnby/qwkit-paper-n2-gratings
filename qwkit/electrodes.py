"""
electrodes.py -- field under periodic alternating-polarity top electrodes over a grounded bottom contact,
and the effective quasi-phase-matching (QPM) coefficient of a field-dependent chi(2).

Geometry: dielectric slab of thickness d between a top electrode plane (y = d) and a ground plane (y = 0),
uniform permittivity (InP, InGaAs and the stack differ by < 12%, neglected). Top potential: segments of
+V and -V of length Lambda/2 - g separated by gaps g in which the potential varies linearly (a standard
model of an un-metallised gap). Exact solution of Laplace's equation as a Fourier sine series:
    phi(x, y) = sum_n b_n sinh(k_n y) / sinh(k_n d) sin(k_n x),  k_n = 2 pi n / Lambda (even-n terms vanish
    for the half-period-antisymmetric pattern; all n are summed numerically)
    E_y(x, y) = -sum_n b_n k_n cosh(k_n y) / sinh(k_n d) sin(k_n x)
QPM coefficient: c_1 = |(1/Lambda) int chi(F(x)) exp(-i 2 pi x / Lambda) dx| (same units as chi); an ideal
sign-reversing square wave of amplitude chi0 gives (2/pi) chi0.
"""
import numpy as np


def top_potential(x, Lam, V, gap):
    """Top-plane potential: +V on (gap/2, Lam/2 - gap/2), -V on (Lam/2 + gap/2, Lam - gap/2), linear ramps."""
    xm = np.mod(x, Lam)
    # distance-based trapezoid: odd about x = Lam/2, period Lam
    out = np.where(xm < Lam / 2, 1.0, -1.0)
    xr = np.where(xm < Lam / 2, xm, xm - Lam / 2)          # position within its half period
    edge = np.minimum(xr, Lam / 2 - xr)                      # distance to the nearest segment end
    ramp = np.clip(edge / (gap / 2.0), 0, 1) if gap > 0 else np.ones_like(edge)
    return V * out * ramp


def field_map(Lam, d, V, gap, y, nx=2048, nmax=401):
    """E_y(x) (V/m) at height y above ground; all lengths in metres. Returns x, Ey."""
    x = np.linspace(0, Lam, nx, endpoint=False)
    phi_top = top_potential(x, Lam, V, gap)
    Ey = np.zeros_like(x)
    for n in range(1, nmax + 1):
        k = 2 * np.pi * n / Lam
        bn = 2.0 / nx * np.sum(phi_top * np.sin(k * x))       # sine coefficient
        if abs(bn) < 1e-12 * V:
            continue
        kd = k * d
        # cosh(ky)/sinh(kd) evaluated stably
        g = np.exp(k * (y - d)) * (1 + np.exp(-2 * k * y)) / (1 - np.exp(-2 * kd))
        Ey += -bn * k * g * np.sin(k * x)
    return x, Ey


def qpm_coefficient(x, chi_x, Lam):
    """|first Fourier coefficient| of chi(x) over one period (complex chi allowed)."""
    c = np.mean(chi_x * np.exp(-1j * 2 * np.pi * x / Lam))
    return abs(c)              # = (2/pi) chi0 for an ideal +-chi0 square wave


def top_potential_array(x, pitch, V, gap, x0=0.0):
    """Top-plane potential of an electrode array: electrode j covers [x0 + j*pitch + gap/2, x0 + (j+1)*pitch - gap/2]
    at potential V[j]; in each gap the potential varies linearly between the neighbouring electrodes; outside the
    array it is 0 (grounded). x in the same length unit as pitch and gap."""
    V = np.asarray(V, float); n = len(V)
    xe = np.concatenate([[x0 - gap / 2], x0 + np.arange(n) * pitch + gap / 2, x0 + (np.arange(n) + 1) * pitch - gap / 2,
                         [x0 + n * pitch + gap / 2]])
    ve = np.concatenate([[0.0], V, V, [0.0]])
    order = np.argsort(xe, kind="stable")
    return np.interp(x, xe[order], ve[order], left=0.0, right=0.0)


def field_profile(x, phi_top, d, y):
    """Vertical field F(x) = -E_y at height y above the ground plane under an arbitrary top-plane potential phi_top(x)
    on a uniform grid x (lengths in m; the grid is treated as periodic, so pad it with grounded margins of a few d).
    Exact Fourier solution of Laplace's equation in the slab; a uniform potential V gives F = V/d."""
    n = len(x); dx = x[1] - x[0]
    k = np.abs(2 * np.pi * np.fft.fftfreq(n, dx))
    ph = np.fft.fft(phi_top)
    g = np.empty(n)
    g[0] = 1.0 / d
    kk = k[1:]
    g[1:] = kk * np.exp(kk * (y - d)) * (1 + np.exp(-2 * kk * y)) / (1 - np.exp(-2 * kk * d))
    return np.real(np.fft.ifft(ph * g))
